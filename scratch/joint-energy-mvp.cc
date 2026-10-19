#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/lte-module.h"
#include "ns3/nr-module.h"
#include "ns3/basic-energy-source-helper.h"
#include "ns3/device-energy-model-container.h"
#include "ns3/energy-source-container.h"
#include "ns3/nr-energy-helper.h"
#include "ns3/nr-ue-drx-model.h"
#include "ns3/nr-ue-energy-model.h"
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <memory>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("JointEnergyMvp");

enum BwpState {
    BWP_NARROW_20MHZ = 0,
    BWP_WIDE_100MHZ  = 1
};

struct BwpResourceProfile {
    uint32_t prbCount;
    double channelBwMhz;
};

const BwpResourceProfile g_bwpResourceProfiles[] = {
    {51,  20.0},
    {273, 100.0}
};

static void
SwitchAllUesToPrimaryBwp(NetDeviceContainer ueDevices, uint8_t targetBwpId)
{
    for (uint32_t i = 0; i < ueDevices.GetN(); ++i)
    {
        Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevices.Get(i));
        NS_ABORT_MSG_IF(!ueDevice || !ueDevice->GetRrc(),
                        "Unable to access UE RRC for physical BWP switch");
        NS_ABORT_MSG_IF(!ueDevice->GetRrc()->RequestPrimaryBwpSwitch(targetBwpId),
                        "UE " << i << " rejected primary BWP switch to "
                              << +targetBwpId);
    }
}

class RlIpcClient
{
public:
    RlIpcClient(const std::string& host = "127.0.0.1", int port = 5555)
        : m_host(host), m_port(port), m_sockFd(-1), m_connected(false)
    {}

    ~RlIpcClient()
    {
        if (m_sockFd >= 0) close(m_sockFd);
    }

    bool Connect()
    {
        m_sockFd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_sockFd < 0) return false;

        struct sockaddr_in servAddr;
        std::memset(&servAddr, 0, sizeof(servAddr));
        servAddr.sin_family = AF_INET;
        servAddr.sin_port = htons(m_port);
        inet_pton(AF_INET, m_host.c_str(), &servAddr.sin_addr);

        if (connect(m_sockFd, (struct sockaddr*)&servAddr, sizeof(servAddr)) < 0)
        {
            close(m_sockFd);
            m_sockFd = -1;
            return false;
        }
        m_connected = true;
        return true;
    }

    int StepIpc(double backlogBytes, double maxUeBacklogBytes, double offeredRateMbps,
                double deliveredRateMbps,
                double delayMs,
                double avgCqi, double p10Cqi, double minCqi, double avgMcs,
                double wActiveMs, int bwpState,
                double gnbPowerW, double uePowerMw, double jainFairness)
    {
        if (!m_connected && !Connect())
        {
            return -1;
        }

        char sendBuffer[384];
        std::snprintf(sendBuffer, sizeof(sendBuffer),
                      "S4,%.2f,%.2f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.4f\n",
                      backlogBytes, maxUeBacklogBytes, offeredRateMbps, deliveredRateMbps, delayMs,
                      avgCqi, p10Cqi, minCqi,
                      avgMcs, wActiveMs, bwpState, gnbPowerW, uePowerMw, jainFairness);

        if (send(m_sockFd, sendBuffer, std::strlen(sendBuffer), 0) < 0)
        {
            m_connected = false;
            return -1;
        }

        char recvBuffer[64];
        std::memset(recvBuffer, 0, sizeof(recvBuffer));
        ssize_t bytesRead = recv(m_sockFd, recvBuffer, sizeof(recvBuffer) - 1, 0);
        if (bytesRead <= 0)
        {
            m_connected = false;
            return -1;
        }

        recvBuffer[bytesRead] = '\0';
        int actionIdx = -1;
        std::sscanf(recvBuffer, "%d", &actionIdx);
        return actionIdx;
    }

private:
    std::string m_host;
    int m_port;
    int m_sockFd;
    bool m_connected;
};

class PacketDelayTracker
{
public:
    PacketDelayTracker()
        : m_totalTxBytes(0),
          m_totalRxBytes(0),
          m_totalDropBytes(0)
    {}

    void RecordTx(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface)
    {
        uint32_t payloadBytes = packet->GetSize() >= 8 ? packet->GetSize() - 8 : packet->GetSize();
        m_txInfo[packet->GetUid()] = {Simulator::Now(), payloadBytes};
        m_pendingBytes += payloadBytes;
        m_totalTxBytes += payloadBytes;
        m_windowTxBytes += payloadBytes;
    }

    void RecordRx(Ptr<const Packet> packet, const Address& from, const Address& to)
    {
        uint64_t uid = packet->GetUid();
        auto it = m_txInfo.find(uid);
        if (it != m_txInfo.end())
        {
            Time sendTime = it->second.first;
            Time delay = Simulator::Now() - sendTime;
            double rawDelayMs = delay.GetSeconds() * 1000.0;
            m_delaysMs.push_back(rawDelayMs);
            m_windowDelaysMs.push_back(rawDelayMs);
            m_pendingBytes = m_pendingBytes >= it->second.second
                                 ? m_pendingBytes - it->second.second
                                 : 0;
            m_txInfo.erase(it);
        }
        m_totalRxBytes += packet->GetSize();
        m_windowRxBytes += packet->GetSize();
    }

    void RecordDrop(Ptr<const Packet> packet)
    {
        m_totalDropBytes += packet->GetSize();
        auto it = m_txInfo.find(packet->GetUid());
        if (it != m_txInfo.end())
        {
            m_pendingBytes = m_pendingBytes >= it->second.second
                                 ? m_pendingBytes - it->second.second
                                 : 0;
            m_txInfo.erase(it);
        }
    }

    uint64_t GetPendingBytes() const
    {
        return m_pendingBytes;
    }

    uint64_t GetTotalTxBytes() const
    {
        return m_totalTxBytes;
    }

    uint64_t GetTotalRxBytes() const
    {
        return m_totalRxBytes;
    }

    double GetP99Latency() const
    {
        return CalculateP99(m_delaysMs);
    }

    double GetWindowP99Latency() const
    {
        return CalculateP99(m_windowDelaysMs);
    }

    void ResetWindowStats()
    {
        m_windowDelaysMs.clear();
        m_windowTxBytes = 0;
        m_windowRxBytes = 0;
    }

    double GetWindowOfferedRateMbps(double windowSeconds) const
    {
        return windowSeconds > 0.0
                   ? (static_cast<double>(m_windowTxBytes) * 8.0) /
                         (windowSeconds * 1e6)
                   : 0.0;
    }

    double GetWindowDeliveredRateMbps(double windowSeconds) const
    {
        return windowSeconds > 0.0
                   ? (static_cast<double>(m_windowRxBytes) * 8.0) /
                         (windowSeconds * 1e6)
                   : 0.0;
    }

    double GetQosSatisfactionRate(double targetDelayMs = 20.0) const
    {
        if (m_delaysMs.empty()) return 0.0;
        uint64_t count = 0;
        for (double d : m_delaysMs)
        {
            if (d <= targetDelayMs) count++;
        }
        return (static_cast<double>(count) / m_delaysMs.size()) * 100.0;
    }

private:
    static double CalculateP99(const std::vector<double>& delaysMs)
    {
        if (delaysMs.empty()) return 0.0;
        std::vector<double> sortedDelays = delaysMs;
        std::sort(sortedDelays.begin(), sortedDelays.end());
        size_t idxP99 = static_cast<size_t>(0.99 * sortedDelays.size());
        if (idxP99 >= sortedDelays.size()) idxP99 = sortedDelays.size() - 1;
        return sortedDelays[idxP99];
    }

    std::unordered_map<uint64_t, std::pair<Time, uint32_t>> m_txInfo;
    std::vector<double> m_delaysMs;
    std::vector<double> m_windowDelaysMs;
    uint64_t m_pendingBytes{0};
    uint64_t m_windowTxBytes{0};
    uint64_t m_windowRxBytes{0};
    uint64_t m_totalTxBytes;
    uint64_t m_totalRxBytes;
    uint64_t m_totalDropBytes;
};

class RadioResourceController
{
public:
    RadioResourceController(Ptr<NrMacSchedulerNs3> scheduler,
                            Ptr<NrGnbPhy> gnbPhy,
                            double periodMs,
                            bool enableGating,
                            bool enableBwpDynamic)
        : m_scheduler(scheduler),
          m_gnbPhy(gnbPhy),
          m_periodMs(periodMs),
          m_enableGating(enableGating),
          m_enableBwpDynamic(enableBwpDynamic),
          m_activeWindowMs(periodMs),
          m_bwp(BWP_WIDE_100MHZ)
    {
        NS_ABORT_MSG_IF(!m_scheduler || !m_gnbPhy,
                        "The gNB scheduler or PHY does not support resource control");
    }

    void Start(double activeWindowMs, BwpState bwp)
    {
        UpdateState(activeWindowMs, bwp);
        Simulator::ScheduleNow(&RadioResourceController::BeginCycle, this);
    }

    void UpdateState(double activeWindowMs, BwpState bwp)
    {
        m_activeWindowMs = std::max(0.0, std::min(m_periodMs, activeWindowMs));
        m_bwp = bwp;
    }

    uint64_t GetDlRlcBacklogBytes() const
    {
        return m_scheduler->GetTotalDlBufferBytes();
    }

    NrMacSchedulerNs3::DlObservationStats GetDlObservationStats() const
    {
        return m_scheduler->GetDlObservationStats();
    }

private:
    void BeginCycle()
    {
        ApplyBwpMask();
        m_scheduler->SetDlDataSchedulingEnabled(true);
        if (m_enableGating && m_activeWindowMs < m_periodMs)
        {
            Simulator::Schedule(MilliSeconds(m_activeWindowMs),
                                &RadioResourceController::EndActiveWindow,
                                this);
        }
        Simulator::Schedule(MilliSeconds(m_periodMs),
                            &RadioResourceController::BeginCycle,
                            this);
    }

    void EndActiveWindow()
    {
        m_scheduler->SetDlDataSchedulingEnabled(false);
    }

    void ApplyBwpMask()
    {
        uint64_t rbPerRbg = m_scheduler->GetNumRbPerRbg();
        NS_ABORT_MSG_IF(rbPerRbg == 0, "The scheduler RBG size is not configured");
        uint32_t totalRbgs = m_gnbPhy->GetRbNum() / rbPerRbg;
        NS_ABORT_MSG_IF(totalRbgs == 0, "The configured PHY has no schedulable RBGs");

        uint32_t usableRbgs = totalRbgs;
        if (m_bwp == BWP_NARROW_20MHZ)
        {
            uint32_t requestedPrbs = g_bwpResourceProfiles[m_bwp].prbCount;
            usableRbgs = std::min(totalRbgs,
                                  static_cast<uint32_t>(std::ceil(
                                      static_cast<double>(requestedPrbs) / rbPerRbg)));
        }
        std::vector<bool> mask(totalRbgs, false);
        uint32_t firstUsable = (totalRbgs - usableRbgs) / 2;
        std::fill(mask.begin() + firstUsable,
                  mask.begin() + firstUsable + usableRbgs,
                  true);
        m_scheduler->SetDlNotchedRbgMask(mask);
    }

    Ptr<NrMacSchedulerNs3> m_scheduler;
    Ptr<NrGnbPhy> m_gnbPhy;
    double m_periodMs;
    bool m_enableGating;
    bool m_enableBwpDynamic;
    double m_activeWindowMs;
    BwpState m_bwp;
};

class OfficialEnergyMetrics;

class DynamicDtxController
{
public:
    DynamicDtxController(PacketDelayTracker* tracker, OfficialEnergyMetrics* energyMetrics,
                         RadioResourceController* radioController, double totalDataRateMbps,
                         bool enableDciDynamic = false,
                         bool enableBwpDynamic = false, bool enableRlIpc = false, int ipcPort = 5555,
                         double linkDelaySec = 0.010, double evalIntervalMs = 40.0, double minActiveMs = 10.0,
                         double maxActiveMs = 30.0, double initialActiveMs = 30.0,
                         double stepUpMs = 4.0, double stepDownMs = 2.0,
                         uint64_t backlogThresholdBytes = 2000,
                         std::string timelineCsvName = "active_window_timeline.csv",
                         double warmupMs = 200.0)
        : m_tracker(tracker),
          m_energyMetrics(energyMetrics),
          m_radioController(radioController),
          m_totalDataRateMbps(totalDataRateMbps),
          m_enableDciDynamic(enableDciDynamic),
          m_enableBwpDynamic(enableBwpDynamic),
          m_enableRlIpc(enableRlIpc),
          m_ipcPort(ipcPort),
          m_evalIntervalSec(evalIntervalMs / 1000.0),
          m_minActiveMs(minActiveMs),
          m_maxActiveMs(maxActiveMs),
          m_stepUpMs(stepUpMs),
          m_stepDownMs(stepDownMs),
          m_currentActiveMs(std::max(minActiveMs, std::min(maxActiveMs, initialActiveMs))),
          m_currentBwp(BWP_NARROW_20MHZ),
          m_bwpInactivityTimerMs(0.0),
          m_bwpInactivityTimeoutMs(10.0),
          m_cellDtxPeriodMs(40.0),
          m_avgTbsPerSlot(3000),
          m_slotDurationMs(0.5),
          m_backlogThresholdBytes(backlogThresholdBytes),
          m_timelineCsvName(timelineCsvName),
          m_warmupMs(warmupMs),
          m_ipcClient(nullptr)
    {
        m_bdpThresholdBytes = static_cast<uint64_t>((totalDataRateMbps * 1e6 * linkDelaySec) / 8.0);
        if (m_enableRlIpc)
        {
            m_ipcClient = new RlIpcClient("127.0.0.1", ipcPort);
        }
    }

    ~DynamicDtxController()
    {
        if (m_ipcClient) delete m_ipcClient;
    }

    void Start()
    {
        bool exists = false;
        {
            std::ifstream f(m_timelineCsvName);
            if (f.good()) exists = true;
        }
        std::ofstream csv(m_timelineCsvName, std::ios::app);
        if (csv.is_open() && !exists)
        {
            csv << "SimTimeSec,ActiveWindowMs,CurrentBwp,TotalPendingBytes,RlcBacklogBytes,DataRateMbps\n";
        }
        csv.close();

        Simulator::Schedule(MilliSeconds(m_warmupMs), &DynamicDtxController::BeginControl, this);
    }

    void BeginControl();
    void EvaluateAndAdjust();
    void EvaluateAndAdjustRlIpc();

    double GetCurrentActiveMs() const { return m_currentActiveMs; }
    BwpState GetCurrentBwp() const { return m_currentBwp; }

private:
    void AppendTimeline(uint64_t totalPending, uint64_t rlcBacklogBytes)
    {
        std::ofstream csv(m_timelineCsvName, std::ios::app);
        if (csv.is_open())
        {
            csv << Simulator::Now().GetSeconds() << ","
                << m_currentActiveMs << ","
                << (m_currentBwp == BWP_WIDE_100MHZ ? 100 : 20) << ","
                << totalPending << ","
                << rlcBacklogBytes << ","
                << m_totalDataRateMbps << "\n";
        }
    }

    PacketDelayTracker* m_tracker;
    OfficialEnergyMetrics* m_energyMetrics;
    RadioResourceController* m_radioController;
    double m_totalDataRateMbps;
    bool m_enableDciDynamic;
    bool m_enableBwpDynamic;
    bool m_enableRlIpc;
    int m_ipcPort;
    double m_evalIntervalSec;
    double m_minActiveMs;
    double m_maxActiveMs;
    double m_stepUpMs;
    double m_stepDownMs;
    double m_currentActiveMs;
    BwpState m_currentBwp;
    double m_bwpInactivityTimerMs;
    double m_bwpInactivityTimeoutMs;
    double m_cellDtxPeriodMs;
    uint32_t m_avgTbsPerSlot;
    double m_slotDurationMs;
    uint64_t m_backlogThresholdBytes;
    uint64_t m_bdpThresholdBytes;
    std::string m_timelineCsvName;
    double m_warmupMs;
    RlIpcClient* m_ipcClient;
};

#if 0 // Archived in custom-energy-v4-checkpoint-2026-10-19; never compiled in v5.
class GnbSlotEnergyTracker
{
public:
    GnbSlotEnergyTracker(double slotDurationMs,
                         double p0Active,
                         double pMicroSleep,
                         double pTxMax,
                         double deltaP,
                         double narrowCapacityMbps,
                         double wideCapacityMbps,
                         double wideActivePowerOverheadW,
                         double wideSleepPowerOverheadW,
                         BwpState initialBwp)
        : m_slotDurationSec(slotDurationMs / 1000.0),
          m_p0Active(p0Active),
          m_pMicroSleep(pMicroSleep),
          m_pTxMax(pTxMax),
          m_deltaP(deltaP),
          m_narrowCapacityMbps(narrowCapacityMbps),
          m_wideCapacityMbps(wideCapacityMbps),
          m_wideActivePowerOverheadW(wideActivePowerOverheadW),
          m_wideSleepPowerOverheadW(wideSleepPowerOverheadW),
          m_currentBwp(initialBwp),
          m_totalEnergyJoules(0.0),
          m_windowEnergyJoules(0.0),
          m_activeSlots(0),
          m_sleepSlots(0),
          m_windowSlots(0),
          m_lastTotalRxBytes(0)
    {}

    void UpdateBwpState(BwpState newBwp)
    {
        m_currentBwp = newBwp;
    }

    void TrackSlot(Ptr<FlowMonitor> monitor)
    {
        uint64_t currentRxBytes = 0;
        FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
        for (auto const& stat : stats)
        {
            currentRxBytes += stat.second.rxBytes;
        }

        uint64_t bytesInSlot = (currentRxBytes >= m_lastTotalRxBytes) ? (currentRxBytes - m_lastTotalRxBytes) : 0;
        m_lastTotalRxBytes = currentRxBytes;

        double slotPower = 0.0;
        if (bytesInSlot > 0)
        {
            m_activeSlots++;
            double slotThroughputMbps = (bytesInSlot * 8.0) / (m_slotDurationSec * 1e6);
            double maxCapacityMbps = m_currentBwp == BWP_WIDE_100MHZ
                                         ? m_wideCapacityMbps
                                         : m_narrowCapacityMbps;
            double slotLoad = std::min(1.0, slotThroughputMbps / maxCapacityMbps);
            double bwpOverhead = m_currentBwp == BWP_WIDE_100MHZ
                                     ? m_wideActivePowerOverheadW
                                     : 0.0;
            slotPower = m_p0Active + bwpOverhead + (m_deltaP * m_pTxMax * slotLoad);
        }
        else
        {
            m_sleepSlots++;
            double bwpOverhead = m_currentBwp == BWP_WIDE_100MHZ
                                     ? m_wideSleepPowerOverheadW
                                     : 0.0;
            slotPower = m_pMicroSleep + bwpOverhead;
        }

        m_totalEnergyJoules += slotPower * m_slotDurationSec;
        m_windowEnergyJoules += slotPower * m_slotDurationSec;
        m_windowSlots++;
        Simulator::Schedule(Seconds(m_slotDurationSec), &GnbSlotEnergyTracker::TrackSlot, this, monitor);
    }

    double GetAvgPower() const { return (m_activeSlots + m_sleepSlots > 0) ? m_totalEnergyJoules / ((m_activeSlots + m_sleepSlots) * m_slotDurationSec) : 0.0; }
    double GetWindowAvgPower() const
    {
        return m_windowSlots > 0
                   ? m_windowEnergyJoules / (m_windowSlots * m_slotDurationSec)
                   : 0.0;
    }
    void ResetWindowStats()
    {
        m_windowEnergyJoules = 0.0;
        m_windowSlots = 0;
    }
    double GetTotalEnergy() const { return m_totalEnergyJoules; }

private:
    double m_slotDurationSec;
    double m_p0Active;
    double m_pMicroSleep;
    double m_pTxMax;
    double m_deltaP;
    double m_narrowCapacityMbps;
    double m_wideCapacityMbps;
    double m_wideActivePowerOverheadW;
    double m_wideSleepPowerOverheadW;
    BwpState m_currentBwp;
    double m_totalEnergyJoules;
    double m_windowEnergyJoules;
    uint32_t m_activeSlots;
    uint32_t m_sleepSlots;
    uint32_t m_windowSlots;
    uint64_t m_lastTotalRxBytes;
};

class UeEnergyTracker
{
public:
    UeEnergyTracker(uint32_t numUes, double slotDurationMs, bool enableCdrx, bool enableDciDynamic = false,
                    bool enableBwpDynamic = false, BwpState initialBwp = BWP_NARROW_20MHZ,
                    double phaseOffsetMs = 0.0, double cdrxOnDurationMs = 10.0,
                    double pDeepSleepMw = 1.5)
        : m_numUes(numUes),
          m_slotDurationSec(slotDurationMs / 1000.0),
          m_enableCdrx(enableCdrx),
          m_enableDciDynamic(enableDciDynamic),
          m_enableBwpDynamic(enableBwpDynamic),
          m_phaseOffsetMs(phaseOffsetMs),
          m_cdrxOnDurationMs(cdrxOnDurationMs),
          m_dynamicOnDurationMs(cdrxOnDurationMs),
          m_currentBwp(initialBwp),
          m_pDeepSleepMw(pDeepSleepMw),
          m_totalEnergyJoules(0.0),
          m_windowEnergyJoules(0.0),
          m_windowSlots(0),
          m_lastRxBytesPerUe(numUes, 0)
    {
        m_ueEnergyJoules.resize(numUes, 0.0);
    }

    void UpdateDynamicOnDurationMs(double newOnDurationMs)
    {
        if (m_enableDciDynamic)
        {
            m_dynamicOnDurationMs = newOnDurationMs;
        }
    }

    void UpdateBwpState(BwpState newBwp)
    {
        if (m_enableBwpDynamic && m_currentBwp != newBwp)
        {
            m_currentBwp = newBwp;
        }
    }

    void TrackSlot(Ptr<FlowMonitor> monitor, DynamicDtxController* controller)
    {
        FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
        std::vector<uint64_t> currentRxBytes(m_numUes, 0);

        for (auto const& stat : stats)
        {
            uint32_t flowId = stat.first;
            uint32_t ueIdx = (flowId - 1) % m_numUes;
            currentRxBytes[ueIdx] += stat.second.rxBytes;
        }

        double nowSec = Simulator::Now().GetSeconds();
        double dtxPeriodSec = 0.040;
        double effectiveOnDurationMs = m_enableDciDynamic ? m_dynamicOnDurationMs : m_cdrxOnDurationMs;

        double ueCdrxCycleTimeSec = std::fmod(nowSec - (m_phaseOffsetMs / 1000.0), dtxPeriodSec);
        if (ueCdrxCycleTimeSec < 0) ueCdrxCycleTimeSec += dtxPeriodSec;

        bool inUeOnDuration = (ueCdrxCycleTimeSec <= (effectiveOnDurationMs / 1000.0));
        BwpPowerProfile bwpProfile = g_bwpPowerProfiles[m_currentBwp];

        for (uint32_t i = 0; i < m_numUes; ++i)
        {
            uint64_t bytesInSlot = (currentRxBytes[i] >= m_lastRxBytesPerUe[i]) ? (currentRxBytes[i] - m_lastRxBytesPerUe[i]) : 0;
            m_lastRxBytesPerUe[i] = currentRxBytes[i];

            double uePowerMw = 0.0;
            if (bytesInSlot > 0)
            {
                uePowerMw = bwpProfile.pRxMw;
            }
            else if (!m_enableCdrx)
            {
                uePowerMw = bwpProfile.pMicroSleepMw;
            }
            else
            {
                if (inUeOnDuration)
                {
                    uePowerMw = bwpProfile.pMicroSleepMw;
                }
                else
                {
                    uePowerMw = m_pDeepSleepMw;
                }
            }

            double energyJoules = (uePowerMw / 1000.0) * m_slotDurationSec;
            m_ueEnergyJoules[i] += energyJoules;
            m_totalEnergyJoules += energyJoules;
            m_windowEnergyJoules += energyJoules;
        }
        m_windowSlots++;

        Simulator::Schedule(Seconds(m_slotDurationSec), &UeEnergyTracker::TrackSlot, this, monitor, controller);
    }

    double GetAvgPowerMwPerUe() const
    {
        double totalTime = Simulator::Now().GetSeconds();
        if (totalTime <= 0 || m_numUes == 0) return 0.0;
        return (m_totalEnergyJoules / totalTime / m_numUes) * 1000.0;
    }

    double GetTotalEnergyJoules() const { return m_totalEnergyJoules; }

    double GetWindowAvgPowerMwPerUe() const
    {
        if (m_windowSlots == 0 || m_numUes == 0) return 0.0;
        double windowDurationSec = m_windowSlots * m_slotDurationSec;
        return (m_windowEnergyJoules / windowDurationSec / m_numUes) * 1000.0;
    }

    void ResetWindowStats()
    {
        m_windowEnergyJoules = 0.0;
        m_windowSlots = 0;
    }

    double GetUeEnergyJoules(uint32_t ueIdx) const
    {
        if (ueIdx < m_ueEnergyJoules.size()) return m_ueEnergyJoules[ueIdx];
        return 0.0;
    }

    double GetBatteryConsumptionMah() const
    {
        return m_totalEnergyJoules / (3.7 * 3.6);
    }

private:
    uint32_t m_numUes;
    double m_slotDurationSec;
    bool m_enableCdrx;
    bool m_enableDciDynamic;
    bool m_enableBwpDynamic;
    double m_phaseOffsetMs;
    double m_cdrxOnDurationMs;
    double m_dynamicOnDurationMs;
    BwpState m_currentBwp;
    double m_pDeepSleepMw;
    double m_totalEnergyJoules;
    double m_windowEnergyJoules;
    uint32_t m_windowSlots;
    std::vector<uint64_t> m_lastRxBytesPerUe;
    std::vector<double> m_ueEnergyJoules;
};
#endif

// Read-only metrics adapter around the official 5G-LENA energy models.  It does
// not estimate power: every value comes from NrGnbEnergyModel/NrUeEnergyModel.
class OfficialEnergyMetrics
{
public:
    OfficialEnergyMetrics(energy::DeviceEnergyModelContainer gnbModels,
                          energy::DeviceEnergyModelContainer ueModels,
                          NodeContainer ueNodes)
        : m_gnbModels(gnbModels),
          m_ueModels(ueModels),
          m_ueNodes(ueNodes),
          m_measurementStart(Seconds(0)),
          m_gnbEnergyBaselineJ(0.0),
          m_ueEnergyBaselineJ(0.0),
          m_windowStart(Seconds(0)),
          m_windowGnbEnergyJ(0.0),
          m_windowUeEnergyJ(0.0)
    {
        m_ueEnergyBaselinesJ.assign(m_ueModels.GetN(), 0.0);
        ResetWindowStats();
    }

    double GetTotalGnbEnergyJ() const
    {
        return GetRawGnbEnergyJ() - m_gnbEnergyBaselineJ;
    }

    double GetRawGnbEnergyJ() const
    {
        double total = 0.0;
        for (uint32_t i = 0; i < m_gnbModels.GetN(); ++i)
        {
            total += m_gnbModels.Get(i)->GetTotalEnergyConsumption();
        }
        return total;
    }

    double GetTotalUeEnergyJ() const
    {
        return GetRawUeEnergyJ() - m_ueEnergyBaselineJ;
    }

    double GetRawUeEnergyJ() const
    {
        double total = 0.0;
        for (uint32_t i = 0; i < m_ueModels.GetN(); ++i)
        {
            total += m_ueModels.Get(i)->GetTotalEnergyConsumption();
        }
        return total;
    }

    double GetAvgGnbPowerW() const
    {
        const double elapsed = (Simulator::Now() - m_measurementStart).GetSeconds();
        return elapsed > 0.0 ? GetTotalGnbEnergyJ() / elapsed : 0.0;
    }

    double GetAvgUePowerMwPerUe() const
    {
        const double elapsed = (Simulator::Now() - m_measurementStart).GetSeconds();
        return elapsed > 0.0 && m_ueModels.GetN() > 0
                   ? GetTotalUeEnergyJ() / elapsed / m_ueModels.GetN() * 1000.0
                   : 0.0;
    }

    double GetWindowAvgGnbPowerW() const
    {
        const double elapsed = (Simulator::Now() - m_windowStart).GetSeconds();
        return elapsed > 0.0
                   ? (GetTotalGnbEnergyJ() - m_windowGnbEnergyJ) / elapsed
                   : 0.0;
    }

    double GetWindowAvgUePowerMwPerUe() const
    {
        const double elapsed = (Simulator::Now() - m_windowStart).GetSeconds();
        return elapsed > 0.0 && m_ueModels.GetN() > 0
                   ? (GetTotalUeEnergyJ() - m_windowUeEnergyJ) / elapsed /
                         m_ueModels.GetN() * 1000.0
                   : 0.0;
    }

    void ResetWindowStats()
    {
        m_windowStart = Simulator::Now();
        m_windowGnbEnergyJ = GetTotalGnbEnergyJ();
        m_windowUeEnergyJ = GetTotalUeEnergyJ();
    }

    void ResetMeasurementBaseline()
    {
        m_measurementStart = Simulator::Now();
        m_gnbEnergyBaselineJ = GetRawGnbEnergyJ();
        m_ueEnergyBaselineJ = GetRawUeEnergyJ();
        for (uint32_t i = 0; i < m_ueModels.GetN(); ++i)
        {
            m_ueEnergyBaselinesJ[i] =
                m_ueModels.Get(i)->GetTotalEnergyConsumption();
        }
        ResetWindowStats();
    }

    void UpdateDynamicOnDurationMs(double onDurationMs)
    {
        for (uint32_t i = 0; i < m_ueNodes.GetN(); ++i)
        {
            Ptr<NrUeDrxModel> drx = m_ueNodes.Get(i)->GetObject<NrUeDrxModel>();
            if (drx)
            {
                drx->SetAttribute("OnDuration", TimeValue(MilliSeconds(onDurationMs)));
            }
        }
    }

    double GetUeEnergyJ(uint32_t ueIndex) const
    {
        return ueIndex < m_ueModels.GetN()
                   ? m_ueModels.Get(ueIndex)->GetTotalEnergyConsumption() -
                         m_ueEnergyBaselinesJ[ueIndex]
                   : 0.0;
    }

    double GetBatteryConsumptionMah(double batteryVoltageV = 3.7) const
    {
        return batteryVoltageV > 0.0
                   ? GetTotalUeEnergyJ() / (batteryVoltageV * 3.6)
                   : 0.0;
    }

private:
    energy::DeviceEnergyModelContainer m_gnbModels;
    energy::DeviceEnergyModelContainer m_ueModels;
    NodeContainer m_ueNodes;
    Time m_measurementStart;
    double m_gnbEnergyBaselineJ;
    double m_ueEnergyBaselineJ;
    std::vector<double> m_ueEnergyBaselinesJ;
    Time m_windowStart;
    double m_windowGnbEnergyJ;
    double m_windowUeEnergyJ;
};

void DynamicDtxController::BeginControl()
{
    if (m_tracker) m_tracker->ResetWindowStats();
    if (m_energyMetrics) m_energyMetrics->ResetWindowStats();

    if (m_enableRlIpc)
    {
        EvaluateAndAdjustRlIpc();
    }
    else
    {
        EvaluateAndAdjust();
    }
}

void DynamicDtxController::EvaluateAndAdjust()
{
    uint64_t totalPending = m_tracker ? m_tracker->GetPendingBytes() : 0;
    uint64_t rlcBacklogBytes = m_radioController
                                   ? m_radioController->GetDlRlcBacklogBytes()
                                   : 0;

    double oldActiveMs = m_currentActiveMs;
    BwpState oldBwp = m_currentBwp;

    if (rlcBacklogBytes > m_backlogThresholdBytes)
    {
        uint64_t excessBacklog = rlcBacklogBytes - m_backlogThresholdBytes;
        uint32_t neededSlots = static_cast<uint32_t>(std::ceil((double)excessBacklog / m_avgTbsPerSlot));
        double targetActiveMs = neededSlots * m_slotDurationMs + m_minActiveMs;
        
        targetActiveMs = std::max(m_minActiveMs, std::min(m_maxActiveMs, targetActiveMs));

        if (targetActiveMs > m_currentActiveMs)
        {
            m_currentActiveMs = std::min(m_maxActiveMs, m_currentActiveMs + m_stepUpMs);
        }
        else if (targetActiveMs < m_currentActiveMs)
        {
            m_currentActiveMs = std::max(m_minActiveMs, m_currentActiveMs - m_stepDownMs);
        }

        if (m_enableBwpDynamic)
        {
            m_currentBwp = BWP_WIDE_100MHZ;
            m_bwpInactivityTimerMs = m_bwpInactivityTimeoutMs;
        }
    }
    else
    {
        m_currentActiveMs = std::max(m_minActiveMs, m_currentActiveMs - m_stepDownMs);

        if (m_enableBwpDynamic)
        {
            if (m_bwpInactivityTimerMs <= 0.0)
            {
                m_currentBwp = BWP_NARROW_20MHZ;
            }
            else
            {
                m_bwpInactivityTimerMs -= (m_evalIntervalSec * 1000.0);
            }
        }
    }

    if (std::abs(m_currentActiveMs - oldActiveMs) > 1e-3 || m_currentBwp != oldBwp)
    {
        if (m_radioController)
        {
            m_radioController->UpdateState(m_currentActiveMs, m_currentBwp);
        }

        if (m_enableDciDynamic)
        {
            if (m_energyMetrics) m_energyMetrics->UpdateDynamicOnDurationMs(m_currentActiveMs);
        }
    }

    AppendTimeline(totalPending, rlcBacklogBytes);
    if (m_tracker) m_tracker->ResetWindowStats();
    if (m_energyMetrics) m_energyMetrics->ResetWindowStats();

    Simulator::Schedule(Seconds(m_evalIntervalSec), &DynamicDtxController::EvaluateAndAdjust, this);
}

void DynamicDtxController::EvaluateAndAdjustRlIpc()
{
    if (!m_ipcClient) return;

    uint64_t totalPending = m_tracker ? m_tracker->GetPendingBytes() : 0;
    NrMacSchedulerNs3::DlObservationStats dlStats;
    if (m_radioController)
    {
        dlStats = m_radioController->GetDlObservationStats();
    }
    uint64_t rlcBacklogBytes = dlStats.totalBufferBytes;
    double offeredRateMbps = m_tracker
                                 ? m_tracker->GetWindowOfferedRateMbps(m_evalIntervalSec)
                                 : 0.0;
    double deliveredRateMbps = m_tracker
                                   ? m_tracker->GetWindowDeliveredRateMbps(m_evalIntervalSec)
                                   : 0.0;
    double p99DelayMs = m_tracker ? m_tracker->GetWindowP99Latency() : 0.0;
    double gnbPowerW = m_energyMetrics ? m_energyMetrics->GetWindowAvgGnbPowerW() : 0.0;
    double uePowerMw = m_energyMetrics ? m_energyMetrics->GetWindowAvgUePowerMwPerUe() : 0.0;
    double jainFairness = 0.74;

    if (m_tracker) m_tracker->ResetWindowStats();
    if (m_energyMetrics) m_energyMetrics->ResetWindowStats();

    int actionIdx = m_ipcClient->StepIpc((double)rlcBacklogBytes,
                                         (double)dlStats.maxUeBufferBytes,
                                         offeredRateMbps,
                                         deliveredRateMbps,
                                         p99DelayMs,
                                         dlStats.meanCqi,
                                         dlStats.p10Cqi,
                                         dlStats.minCqi,
                                         dlStats.meanMcs,
                                         m_currentActiveMs,
                                         (int)m_currentBwp,
                                         gnbPowerW,
                                         uePowerMw,
                                         jainFairness);

    if (actionIdx < 0 || actionIdx >= 10)
    {
        Simulator::Stop();
        return;
    }

    if (actionIdx >= 0)
    {
        double deltaWOptions[] = {-4.0, -2.0, 0.0, 2.0, 4.0};
        double deltaW = deltaWOptions[actionIdx % 5];
        BwpState nextBwp = (actionIdx / 5 == 1) ? BWP_WIDE_100MHZ : BWP_NARROW_20MHZ;

        m_currentActiveMs = std::max(m_minActiveMs, std::min(m_maxActiveMs, m_currentActiveMs + deltaW));
        m_currentBwp = nextBwp;

        if (m_radioController)
        {
            m_radioController->UpdateState(m_currentActiveMs, m_currentBwp);
        }

        if (m_enableDciDynamic)
        {
            if (m_energyMetrics) m_energyMetrics->UpdateDynamicOnDurationMs(m_currentActiveMs);
        }
    }

    AppendTimeline(totalPending, rlcBacklogBytes);

    Simulator::Schedule(Seconds(m_evalIntervalSec), &DynamicDtxController::EvaluateAndAdjustRlIpc, this);
}

void ExportStatsToCsv(const std::string &filename, double simTime, bool enableGating, bool enableQueueAware,
                       double cellDtxPeriodMs, double cellDtxActiveMs, double udpDataRate,
                       uint64_t totalRxBytes, double avgPowerWatts, double totalEnergyJoules,
                       double energyEfficiency, double p99LatencyMs, double qosSatisfactionRate,
                       bool enableCdrx = false, bool enableDciDynamic = false, bool enableBwpDynamic = false,
                       double phaseOffsetMs = 0.0, double ueAvgPowerMw = 0.0, double ueTotalEnergyJoules = 0.0,
                       double ueBatteryMah = 0.0, double ue3EnergyJoules = 0.0)
{
    bool fileExists = false;
    {
        std::ifstream checkFile(filename);
        if (checkFile.good()) fileExists = true;
    }

    std::ofstream csvFile(filename, std::ios::app);
    if (!csvFile.is_open()) return;

    if (!fileExists)
    {
        csvFile << "SimTime,EnableGating,EnableQueueAware,EnableCdrx,EnableDciDynamic,EnableBwpDynamic,PhaseOffsetMs,CellDtxPeriodMs,CellDtxActiveMs,UdpDataRateMbps,TotalRxBytes,TotalRxMbits,AvgPowerWatts,TotalEnergyJoules,EnergyEfficiency_MbitsPerJoule,P99LatencyMs,QosSatisfactionRatePct,UeAvgPowerMw,UeTotalEnergyJoules,UeBatteryMah,Ue3EnergyJoules\n";
    }

    double totalRxMbits = (totalRxBytes * 8.0) / 1e6;

    csvFile << simTime << ","
            << (enableGating ? 1 : 0) << ","
            << (enableQueueAware ? 1 : 0) << ","
            << (enableCdrx ? 1 : 0) << ","
            << (enableDciDynamic ? 1 : 0) << ","
            << (enableBwpDynamic ? 1 : 0) << ","
            << phaseOffsetMs << ","
            << cellDtxPeriodMs << ","
            << cellDtxActiveMs << ","
            << udpDataRate << ","
            << totalRxBytes << ","
            << totalRxMbits << ","
            << avgPowerWatts << ","
            << totalEnergyJoules << ","
            << energyEfficiency << ","
            << p99LatencyMs << ","
            << qosSatisfactionRate << ","
            << ueAvgPowerMw << ","
            << ueTotalEnergyJoules << ","
            << ueBatteryMah << ","
            << ue3EnergyJoules << "\n";

    csvFile.close();
}

int main(int argc, char *argv[])
{
    uint32_t numUes = 10;
    uint32_t numerology = 1;
    double simTime = 10.0;
    double udpPacketRateMbps = 2.0;
    bool enableBurstTraffic = false;
    bool synchronizeBurstTraffic = false;
    double udpBurstRateMbps = 3.0;
    double burstMeanOnMs = 80.0;
    double burstMeanOffMs = 120.0;
    double gnbTxPowerDbm = 30.0;
    std::string gnbEnergyBsCategory = "BsCat1";
    std::string gnbEnergyRefConfig = "Set1";
    std::string gnbEnergyEtaMode = "Dual";
    double gnbEnergyAntennaRatio = 0.4;
    double gnbEnergyPowerUnitW = 1.0;
    double ueEnergyPowerUnitMw = 1.0;
    bool enableGating = true;
    bool enableQueueAware = false;
    bool enableCdrx = false;
    bool enableDciDynamic = false;
    bool enableBwpDynamic = false;
    bool enableTrueBwp = false;
    uint32_t fixedBwpMHz = 100;
    bool enableRlIpc = false;
    int ipcPort = 5555;
    double phaseOffsetMs = 0.0;
    uint32_t cdrxOnDurationMs = 10;
    uint32_t cdrxInactivityMs = 10;
    double cellDtxPeriodMs = 40.0;
    double cellDtxActiveMs = 10.0;
    double qosTargetDelayMs = 20.0;
    double controllerEvalIntervalMs = 40.0;
    double controllerWarmupMs = 200.0;
    double controllerMinActiveMs = 10.0;
    double controllerMaxActiveMs = 30.0;
    double controllerInitialActiveMs = 30.0;
    double controllerStepUpMs = 4.0;
    double controllerStepDownMs = 2.0;
    uint64_t controllerBacklogThresholdBytes = 2000;
    uint32_t rngSeed = 42;
    uint64_t rngRun = 1;
    std::string csvFileName = "joint_energy_results.csv";
    std::string timelineCsvName = "active_window_timeline.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("numUes", "Number of UEs", numUes);
    cmd.AddValue("numerology", "NR numerology (1 gives 30 kHz SCS)", numerology);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("udpDataRate", "Constant/base UDP downlink rate in Mbps per UE", udpPacketRateMbps);
    cmd.AddValue("enableBurstTraffic", "Add an independent On/Off burst source for every UE", enableBurstTraffic);
    cmd.AddValue("synchronizeBurstTraffic", "Use a shared deterministic On/Off burst regime across UEs", synchronizeBurstTraffic);
    cmd.AddValue("udpBurstRate", "Additional UDP burst rate in Mbps per UE", udpBurstRateMbps);
    cmd.AddValue("burstMeanOnMs", "Mean burst On duration in ms", burstMeanOnMs);
    cmd.AddValue("burstMeanOffMs", "Mean burst Off duration in ms", burstMeanOffMs);
    cmd.AddValue("gnbTxPowerDbm", "gNB downlink transmission power in dBm", gnbTxPowerDbm);
    cmd.AddValue("gnbEnergyBsCategory", "TR 38.864 BS category (BsCat1 or BsCat2)", gnbEnergyBsCategory);
    cmd.AddValue("gnbEnergyRefConfig", "TR 38.864 reference set (Set1 for this 100 MHz mu=1 scenario)", gnbEnergyRefConfig);
    cmd.AddValue("gnbEnergyEtaMode", "TR 38.864 PA efficiency mode (Single or Dual)", gnbEnergyEtaMode);
    cmd.AddValue("gnbEnergyAntennaRatio", "TR 38.864 antenna dynamic-power fraction A", gnbEnergyAntennaRatio);
    cmd.AddValue("gnbEnergyPowerUnitW", "Absolute watts per TR 38.864 relative power unit", gnbEnergyPowerUnitW);
    cmd.AddValue("ueEnergyPowerUnitMw", "Absolute mW per TR 38.840 relative power unit", ueEnergyPowerUnitMw);
    cmd.AddValue("enableGating", "Enable Cell-DTX Scheduler Gating", enableGating);
    cmd.AddValue("enableQueueAware", "Enable Closed-Loop Dynamic Controller", enableQueueAware);
    cmd.AddValue("enableCdrx", "Enable UE C-DRX Energy Saving Mode", enableCdrx);
    cmd.AddValue("enableDciDynamic", "Enable 3GPP DCI-Level Dynamic Alignment", enableDciDynamic);
    cmd.AddValue("enableBwpDynamic",
                 "Legacy name: dynamically switch the schedulable PRB profile "
                 "(20/100 MHz-equivalent) on one physical 100 MHz BWP",
                 enableBwpDynamic);
    cmd.AddValue("enableTrueBwp",
                 "Experimental v6: create physical 20 and 100 MHz gNB/UE BWPs; "
                 "fixed-mode validation only",
                 enableTrueBwp);
    cmd.AddValue("fixedBwpMHz",
                 "Fixed schedulable PRB profile (20 or 100 MHz-equivalent); the PHY remains 100 MHz",
                 fixedBwpMHz);
    cmd.AddValue("enableRlIpc", "Enable Socket IPC for Reinforcement Learning Co-Simulation", enableRlIpc);
    cmd.AddValue("ipcPort", "Socket IPC Server Port", ipcPort);
    cmd.AddValue("phaseOffsetMs", "Phase Offset between gNB and UE C-DRX in ms", phaseOffsetMs);
    cmd.AddValue("cdrxOnDurationMs", "C-DRX OnDuration Timer in ms", cdrxOnDurationMs);
    cmd.AddValue("cdrxInactivityMs", "C-DRX Inactivity Timer in ms", cdrxInactivityMs);
    cmd.AddValue("cellDtxPeriodMs", "Cell-DTX Period in ms", cellDtxPeriodMs);
    cmd.AddValue("cellDtxActiveMs", "Cell-DTX Active Duration in ms", cellDtxActiveMs);
    cmd.AddValue("qosTargetDelayMs", "QoS Target Delay Bound in ms", qosTargetDelayMs);
    cmd.AddValue("controllerEvalIntervalMs", "Rule/DRL controller evaluation interval in ms", controllerEvalIntervalMs);
    cmd.AddValue("controllerWarmupMs", "Warm-up time before the first controller state in ms", controllerWarmupMs);
    cmd.AddValue("controllerMinActiveMs", "Minimum controller active window in ms", controllerMinActiveMs);
    cmd.AddValue("controllerMaxActiveMs", "Maximum controller active window in ms", controllerMaxActiveMs);
    cmd.AddValue("controllerInitialActiveMs", "Controller active window at episode start in ms", controllerInitialActiveMs);
    cmd.AddValue("controllerStepUpMs", "Rule-Based active-window increase per decision in ms", controllerStepUpMs);
    cmd.AddValue("controllerStepDownMs", "Rule-Based active-window decrease per decision in ms", controllerStepDownMs);
    cmd.AddValue("controllerBacklogThresholdBytes", "Rule-Based RLC backlog threshold in bytes", controllerBacklogThresholdBytes);
    cmd.AddValue("rngSeed", "ns-3 random-number generator seed", rngSeed);
    cmd.AddValue("rngRun", "ns-3 independent random-number generator run", rngRun);
    cmd.AddValue("csvFileName", "Output CSV file name", csvFileName);
    cmd.AddValue("timelineCsvName", "Controller timeline CSV file name", timelineCsvName);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(fixedBwpMHz != 20 && fixedBwpMHz != 100,
                    "fixedBwpMHz must be either 20 or 100");
    NS_ABORT_MSG_IF(enableTrueBwp && enableBwpDynamic,
                    "v6 phase 1 validates fixed physical BWPs before enabling dynamic switching");
    NS_ABORT_MSG_IF(udpPacketRateMbps <= 0.0, "udpDataRate must be greater than zero");
    NS_ABORT_MSG_IF(enableBurstTraffic && udpBurstRateMbps <= 0.0,
                    "udpBurstRate must be greater than zero when burst traffic is enabled");
    NS_ABORT_MSG_IF(enableBurstTraffic &&
                        (burstMeanOnMs <= 0.0 || burstMeanOffMs <= 0.0),
                    "burst On/Off means must be greater than zero");
    NS_ABORT_MSG_IF(controllerInitialActiveMs < controllerMinActiveMs ||
                        controllerInitialActiveMs > controllerMaxActiveMs,
                    "controllerInitialActiveMs must be inside the controller min/max range");
    NS_ABORT_MSG_IF(gnbEnergyPowerUnitW <= 0.0 || ueEnergyPowerUnitMw <= 0.0,
                    "Official Energy model PowerUnit values must be greater than zero");
    NS_ABORT_MSG_IF(gnbEnergyAntennaRatio < 0.0 || gnbEnergyAntennaRatio > 1.0,
                    "gnbEnergyAntennaRatio must be inside [0, 1]");
    NS_ABORT_MSG_IF(enableCdrx && phaseOffsetMs != 0.0,
                    "NrEnergyHelper v5 does not expose a DRX start phase offset; use phaseOffsetMs=0");

    RngSeedManager::SetSeed(rngSeed);
    RngSeedManager::SetRun(rngRun);

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetEpcHelper(epcHelper);

    NodeContainer gnbNodes;
    NodeContainer ueNodes;
    gnbNodes.Create(1);
    ueNodes.Create(numUes);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    
    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    gnbPositionAlloc->Add(Vector(0.0, 0.0, 10.0));
    mobility.SetPositionAllocator(gnbPositionAlloc);
    mobility.Install(gnbNodes);

    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < numUes; ++i)
    {
        double dist = 30.0 + (i * 12.0);
        uePositionAlloc->Add(Vector(dist, 0.0, 1.5));
    }
    mobility.SetPositionAllocator(uePositionAlloc);
    mobility.Install(ueNodes);

    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);

    CcBwpCreator ccBwpCreator;
    OperationBandInfo band;
    if (!enableTrueBwp)
    {
        CcBwpCreator::SimpleOperationBandConf bandConf(3.5e9, 100e6, 1);
        band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    }
    else
    {
        // 5G NR allows a narrow BWP to be contained in a wider carrier BWP.
        // CcBwpCreator partitions BWPs into non-overlapping slices, so the v6
        // prototype describes the nested pair explicitly.  Each entry creates
        // a real gNB/UE PHY with its own channel bandwidth; fixed-mode tests
        // must pass before any runtime switching policy is connected.
        constexpr double centerHz = 3.5e9;
        constexpr double wideBandwidthHz = 100e6;
        constexpr double narrowBandwidthHz = 20e6;
        constexpr double narrowCenterHz = centerHz - 40e6;
        band.m_centralFrequency = centerHz;
        band.m_channelBandwidth = wideBandwidthHz;
        band.m_lowerFrequency = centerHz - wideBandwidthHz / 2.0;
        band.m_higherFrequency = centerHz + wideBandwidthHz / 2.0;

        auto cc = std::make_unique<ComponentCarrierInfo>();
        cc->m_ccId = 0;
        cc->m_centralFrequency = centerHz;
        cc->m_channelBandwidth = wideBandwidthHz;
        cc->m_lowerFrequency = band.m_lowerFrequency;
        cc->m_higherFrequency = band.m_higherFrequency;

        auto narrowBwp = std::make_unique<BandwidthPartInfo>();
        narrowBwp->m_bwpId = BWP_NARROW_20MHZ;
        narrowBwp->m_centralFrequency = narrowCenterHz;
        narrowBwp->m_channelBandwidth = narrowBandwidthHz;
        narrowBwp->m_lowerFrequency = narrowCenterHz - narrowBandwidthHz / 2.0;
        narrowBwp->m_higherFrequency = narrowCenterHz + narrowBandwidthHz / 2.0;
        cc->m_bwp.emplace_back(std::move(narrowBwp));

        auto wideBwp = std::make_unique<BandwidthPartInfo>();
        wideBwp->m_bwpId = BWP_WIDE_100MHZ;
        wideBwp->m_centralFrequency = centerHz;
        wideBwp->m_channelBandwidth = wideBandwidthHz;
        wideBwp->m_lowerFrequency = band.m_lowerFrequency;
        wideBwp->m_higherFrequency = band.m_higherFrequency;
        cc->m_bwp.emplace_back(std::move(wideBwp));
        band.m_cc.emplace_back(std::move(cc));
    }

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    channelHelper->AssignChannelsToBands({band});

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    const uint8_t selectedBwpId = enableTrueBwp && fixedBwpMHz == 100
                                      ? BWP_WIDE_100MHZ
                                      : BWP_NARROW_20MHZ;
    if (enableTrueBwp)
    {
        nrHelper->SetGnbBwpManagerAlgorithmAttribute(
            "NGBR_VIDEO_TCP_DEFAULT", UintegerValue(selectedBwpId));
        nrHelper->SetUeBwpManagerAlgorithmAttribute(
            "NGBR_VIDEO_TCP_DEFAULT", UintegerValue(selectedBwpId));
    }

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);

    if (enableTrueBwp)
    {
        for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
        {
            DynamicCast<NrUeNetDevice>(ueNetDev.Get(ueId))
                ->GetRrc()
                ->SetAttribute("EnableAutomaticBwpSwitch", BooleanValue(false));
        }
    }

    Ptr<NrMacSchedulerNs3> scheduler =
        DynamicCast<NrMacSchedulerNs3>(NrHelper::GetScheduler(gnbNetDev.Get(0), selectedBwpId));
    Ptr<NrGnbPhy> gnbPhy = NrHelper::GetGnbPhy(gnbNetDev.Get(0), selectedBwpId);
    NS_ABORT_MSG_IF(!scheduler || !gnbPhy, "Unable to access the gNB scheduler or PHY");
    for (uint32_t bwpId = 0; bwpId < allBwps.size(); ++bwpId)
    {
        Ptr<NrGnbPhy> configuredPhy = NrHelper::GetGnbPhy(gnbNetDev.Get(0), bwpId);
        configuredPhy->SetAttribute("Numerology", UintegerValue(numerology));
        configuredPhy->SetAttribute("TxPower", DoubleValue(gnbTxPowerDbm));
        for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
        {
            // Secondary UE BWPs have not necessarily received their RRC PHY
            // configuration when the Energy listener is installed.  Give all
            // physical BWPs the scenario numerology up front so their slot
            // duration is valid independently of which BWP starts primary.
            NrHelper::GetUePhy(ueNetDev.Get(ueId), bwpId)
                ->SetNumerology(numerology);
        }
    }
    RadioResourceController radioController(scheduler,
                                            gnbPhy,
                                            cellDtxPeriodMs,
                                            enableGating,
                                            enableBwpDynamic);

    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);
    internet.Install(ueNodes);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.010)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    PacketDelayTracker delayTracker;
    Ptr<Ipv4L3Protocol> remoteHostIpv4 = remoteHost->GetObject<Ipv4L3Protocol>();
    if (remoteHostIpv4)
    {
        remoteHostIpv4->TraceConnectWithoutContext("Tx", MakeCallback(&PacketDelayTracker::RecordTx, &delayTracker));
    }

    Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/$ns3::NrGnbNetDevice/LteRlc/TxDrop",
        MakeCallback(&PacketDelayTracker::RecordDrop, &delayTracker)
    );

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDev));

    for (uint32_t i = 0; i < ueNetDev.GetN(); ++i)
    {
        nrHelper->AttachToGnb(ueNetDev.Get(i), gnbNetDev.Get(0));
    }

    uint16_t port = 1234;
    ApplicationContainer clientApps;
    ApplicationContainer serverApps;

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        UdpServerHelper server(port + i);
        ApplicationContainer sApps = server.Install(ueNodes.Get(i));
        serverApps.Add(sApps);

        Ptr<UdpServer> udpServer = DynamicCast<UdpServer>(sApps.Get(0));
        if (udpServer)
        {
            udpServer->TraceConnectWithoutContext("RxWithAddresses", MakeCallback(&PacketDelayTracker::RecordRx, &delayTracker));
        }

        const InetSocketAddress destination(ueIpIface.GetAddress(i), port + i);
        OnOffHelper baseTraffic("ns3::UdpSocketFactory", destination);
        baseTraffic.SetAttribute("PacketSize", UintegerValue(1024));
        baseTraffic.SetAttribute(
            "DataRate",
            DataRateValue(DataRate(std::to_string(udpPacketRateMbps) + "Mbps")));
        baseTraffic.SetAttribute(
            "OnTime",
            StringValue("ns3::ConstantRandomVariable[Constant=100.0]"));
        baseTraffic.SetAttribute(
            "OffTime",
            StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
        clientApps.Add(baseTraffic.Install(remoteHost));

        if (enableBurstTraffic)
        {
            OnOffHelper burstTraffic("ns3::UdpSocketFactory", destination);
            burstTraffic.SetAttribute("PacketSize", UintegerValue(1024));
            burstTraffic.SetAttribute(
                "DataRate",
                DataRateValue(DataRate(std::to_string(udpBurstRateMbps) + "Mbps")));
            if (synchronizeBurstTraffic)
            {
                burstTraffic.SetAttribute(
                    "OnTime",
                    StringValue("ns3::ConstantRandomVariable[Constant=" +
                                std::to_string(burstMeanOnMs / 1000.0) + "]"));
                burstTraffic.SetAttribute(
                    "OffTime",
                    StringValue("ns3::ConstantRandomVariable[Constant=" +
                                std::to_string(burstMeanOffMs / 1000.0) + "]"));
            }
            else
            {
                burstTraffic.SetAttribute(
                    "OnTime",
                    StringValue("ns3::ExponentialRandomVariable[Mean=" +
                                std::to_string(burstMeanOnMs / 1000.0) + "]"));
                burstTraffic.SetAttribute(
                    "OffTime",
                    StringValue("ns3::ExponentialRandomVariable[Mean=" +
                                std::to_string(burstMeanOffMs / 1000.0) + "]"));
            }
            clientApps.Add(burstTraffic.Install(remoteHost));
        }
    }

    serverApps.Start(Seconds(0.01));
    // Warm the channel and attachment state without creating packets that the
    // controller could not influence. Traffic and control begin together.
    const double trafficStartSec = std::max(0.02, controllerWarmupMs / 1000.0);
    if (enableTrueBwp && selectedBwpId != BWP_NARROW_20MHZ)
    {
        const double switchTimeSec = std::max(0.01, trafficStartSec - 0.05);
        Simulator::Schedule(Seconds(switchTimeSec),
                            &SwitchAllUesToPrimaryBwp,
                            ueNetDev,
                            selectedBwpId);
    }
    clientApps.Start(Seconds(trafficStartSec));

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    BwpState initialBwp = enableBwpDynamic || fixedBwpMHz == 20
                              ? BWP_NARROW_20MHZ
                              : BWP_WIDE_100MHZ;

    // Official 5G-LENA energy framework: PHY traces drive the TR 38.864 gNB
    // and TR 38.840 UE models at the real OFDM symbol/slot timing.  No custom
    // periodic energy-accounting event is scheduled in this scenario.
    BasicEnergySourceHelper energySourceHelper;
    energySourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(1e9));
    energy::EnergySourceContainer gnbEnergySources = energySourceHelper.Install(gnbNodes);
    energy::EnergySourceContainer ueEnergySources = energySourceHelper.Install(ueNodes);

    NrEnergyHelper nrEnergyHelper;
    nrEnergyHelper.SetGnbEnergyModelAttribute("BsCategory", StringValue(gnbEnergyBsCategory));
    nrEnergyHelper.SetGnbEnergyModelAttribute("RefConfigSet", StringValue(gnbEnergyRefConfig));
    nrEnergyHelper.SetGnbEnergyModelAttribute("EtaMode", StringValue(gnbEnergyEtaMode));
    nrEnergyHelper.SetGnbEnergyModelAttribute("AntennaRatio_A",
                                               DoubleValue(gnbEnergyAntennaRatio));
    nrEnergyHelper.SetGnbEnergyModelAttribute("PowerUnit",
                                               DoubleValue(gnbEnergyPowerUnitW));
    nrEnergyHelper.SetGnbEnergyModelAttribute("ReferenceRbCount",
                                               UintegerValue(enableTrueBwp
                                                                 ? NrHelper::GetGnbPhy(
                                                                       gnbNetDev.Get(0),
                                                                       BWP_WIDE_100MHZ)
                                                                       ->GetRbNum()
                                                                 : gnbPhy->GetRbNum()));
    nrEnergyHelper.SetUeEnergyModelAttribute("FreqRange", StringValue("FR1"));
    nrEnergyHelper.SetUeEnergyModelAttribute("PowerUnit", DoubleValue(ueEnergyPowerUnitMw));
    nrEnergyHelper.SetUeEnergyModelAttribute("ReferenceBwpBandwidth", UintegerValue(100));
    nrEnergyHelper.EnableDrx(enableCdrx);
    nrEnergyHelper.SetDrxModelAttribute("ReferenceConfig", StringValue("CUSTOM"));
    nrEnergyHelper.SetDrxModelAttribute("LongCycle", TimeValue(MilliSeconds(cellDtxPeriodMs)));
    nrEnergyHelper.SetDrxModelAttribute("OnDuration", TimeValue(MilliSeconds(cdrxOnDurationMs)));
    nrEnergyHelper.SetDrxModelAttribute("InactivityTimer",
                                        TimeValue(MilliSeconds(cdrxInactivityMs)));

    energy::DeviceEnergyModelContainer gnbEnergyModels =
        nrEnergyHelper.InstallGnb(gnbNetDev, gnbEnergySources);
    energy::DeviceEnergyModelContainer ueEnergyModels =
        nrEnergyHelper.InstallUe(ueNetDev, ueEnergySources, selectedBwpId);
    OfficialEnergyMetrics energyMetrics(gnbEnergyModels, ueEnergyModels, ueNodes);
    if (enableTrueBwp)
    {
        Simulator::Schedule(Seconds(trafficStartSec),
                            &OfficialEnergyMetrics::ResetMeasurementBaseline,
                            &energyMetrics);
    }

    const double burstDutyCycle = enableBurstTraffic
                                      ? burstMeanOnMs / (burstMeanOnMs + burstMeanOffMs)
                                      : 0.0;
    const double meanRatePerUeMbps =
        udpPacketRateMbps + burstDutyCycle * udpBurstRateMbps;
    double totalTrafficRateMbps = meanRatePerUeMbps * numUes;
    DynamicDtxController dynamicController(&delayTracker, &energyMetrics,
                                           &radioController,
                                           totalTrafficRateMbps, enableDciDynamic, enableBwpDynamic,
                                           enableRlIpc, ipcPort, 0.010, controllerEvalIntervalMs,
                                           controllerMinActiveMs, controllerMaxActiveMs,
                                           controllerInitialActiveMs,
                                           controllerStepUpMs, controllerStepDownMs,
                                           controllerBacklogThresholdBytes, timelineCsvName,
                                           controllerWarmupMs);
    bool dynamicControlEnabled = enableGating && (enableQueueAware || enableRlIpc);
    radioController.Start(cellDtxActiveMs, initialBwp);
    if (dynamicControlEnabled)
    {
        dynamicController.Start();
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer flowStats = monitor->GetFlowStats();
    uint64_t totalRxBytes = 0;
    for (auto iter = flowStats.begin(); iter != flowStats.end(); ++iter)
    {
        totalRxBytes += iter->second.rxBytes;
    }

    double finalActiveMs = dynamicControlEnabled ? dynamicController.GetCurrentActiveMs() : cellDtxActiveMs;
    double avgPower = energyMetrics.GetAvgGnbPowerW();
    double totalEnergy = energyMetrics.GetTotalGnbEnergyJ();
    double totalRxMbits = (totalRxBytes * 8.0) / 1e6;
    double energyEff = (totalEnergy > 0) ? (totalRxMbits / totalEnergy) : 0.0;
    double p99Latency = delayTracker.GetP99Latency();
    double qosSatisfaction = delayTracker.GetQosSatisfactionRate(qosTargetDelayMs);

    std::cout << "OfferedTxBytes=" << delayTracker.GetTotalTxBytes()
              << ", PendingBytes=" << delayTracker.GetPendingBytes()
              << ", AppRxBytes=" << delayTracker.GetTotalRxBytes() << std::endl;

    double ueAvgPowerMw = energyMetrics.GetAvgUePowerMwPerUe();
    double ueTotalEnergy = energyMetrics.GetTotalUeEnergyJ();
    double ueBatteryMah = energyMetrics.GetBatteryConsumptionMah();
    double ue3EnergyJoules = energyMetrics.GetUeEnergyJ(2);

    std::cout << "EnergyModel=5G-LENA-Official-TR38.864/TR38.840"
              << ", GnbRefConfig=" << gnbEnergyRefConfig
              << ", GnbBsCategory=" << gnbEnergyBsCategory
              << ", GnbPowerUnitW=" << gnbEnergyPowerUnitW
              << ", UePowerUnitMw=" << ueEnergyPowerUnitMw
              << ", FrequencyControl="
              << (enableTrueBwp ? "fixed-physical-BWP-v6-prototype"
                                : "PRB-profile-on-100MHz-PHY")
              << ", ActivePhysicalBwpMHz="
              << (enableTrueBwp ? fixedBwpMHz : 100) << std::endl;

    ExportStatsToCsv(csvFileName, Simulator::Now().GetSeconds(), enableGating, enableQueueAware, cellDtxPeriodMs, finalActiveMs, udpPacketRateMbps, totalRxBytes, avgPower, totalEnergy, energyEff, p99Latency, qosSatisfaction, enableCdrx, enableDciDynamic, enableBwpDynamic, phaseOffsetMs, ueAvgPowerMw, ueTotalEnergy, ueBatteryMah, ue3EnergyJoules);

    Simulator::Destroy();
    return 0;
}
