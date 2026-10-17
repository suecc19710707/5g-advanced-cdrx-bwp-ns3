#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/lte-module.h"
#include "ns3/nr-module.h"
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("JointEnergyMvp");

// ==================================================================================
// 1. 3GPP BWP 規格與動態切換功率映射表 (3GPP TS 38.104 / TS 38.133 / TR 38.840)
// ==================================================================================
enum BwpState {
    BWP_NARROW_20MHZ = 0,  // 20 MHz Low-Power BWP (51 PRBs, 30kHz SCS)
    BWP_WIDE_100MHZ  = 1   // 100 MHz High-Performance BWP (273 PRBs, 30kHz SCS)
};

struct BwpPowerProfile {
    double pRxMw;          // Active Rx 接收解碼功耗 (mW)
    double pMicroSleepMw;  // Active Idle / Micro-Sleep PDCCH 聽診功耗 (mW)
    uint32_t prbCount;     // PRB 數量
    double channelBwMhz;   // 頻道頻寬 (MHz)
};

// 3GPP BWP 實體層功率參數 (20MHz vs 100MHz)
const BwpPowerProfile g_bwpPowerProfiles[] = {
    {120.0, 20.0, 51,  20.0},  // Index 0: 20 MHz BWP (BWP_NARROW_20MHZ)
    {250.0, 45.0, 273, 100.0}  // Index 1: 100 MHz BWP (BWP_WIDE_100MHZ)
};

// ----------------------------------------------------------------------------------
// 2. 封包延遲與佇列追蹤器 (支援 DCI 動態 OnDuration 擴展與喚醒延遲懲罰)
// ----------------------------------------------------------------------------------
class PacketDelayTracker
{
public:
    PacketDelayTracker(bool enableCdrx = false, bool enableDciDynamic = false, double phaseOffsetMs = 0.0, double cdrxOnDurationMs = 10.0, double cycleMs = 40.0)
        : m_enableCdrx(enableCdrx),
          m_enableDciDynamic(enableDciDynamic),
          m_phaseOffsetMs(phaseOffsetMs),
          m_cdrxOnDurationMs(cdrxOnDurationMs),
          m_dynamicOnDurationMs(cdrxOnDurationMs),
          m_cycleMs(cycleMs),
          m_totalTxBytes(0),
          m_totalRxBytes(0),
          m_totalDropBytes(0)
    {}

    void UpdateDynamicOnDurationMs(double newOnDurationMs)
    {
        if (m_enableDciDynamic)
        {
            m_dynamicOnDurationMs = newOnDurationMs;
        }
    }

    void RecordTx(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface)
    {
        m_txTimes[packet->GetUid()] = Simulator::Now();
        m_totalTxBytes += packet->GetSize();
    }

    void RecordRx(Ptr<const Packet> packet, const Address& from, const Address& to)
    {
        uint64_t uid = packet->GetUid();
        auto it = m_txTimes.find(uid);
        if (it != m_txTimes.end())
        {
            Time sendTime = it->second;
            Time delay = Simulator::Now() - sendTime;
            double rawDelayMs = delay.GetSeconds() * 1000.0;

            // 硬性 C-DRX PHY 門鎖判定
            if (m_enableCdrx)
            {
                double effectiveOnDurationMs = m_enableDciDynamic ? m_dynamicOnDurationMs : m_cdrxOnDurationMs;
                double sendTimeSec = sendTime.GetSeconds();
                double cycleTimeMs = std::fmod(sendTimeSec * 1000.0, m_cycleMs);

                double ueOnStart = m_phaseOffsetMs;
                double ueOnEnd = m_phaseOffsetMs + effectiveOnDurationMs;

                // 若發射時間非屬於 UE 甦醒視窗，則累加喚醒等待延遲懲罰
                if (cycleTimeMs < ueOnStart || cycleTimeMs > ueOnEnd)
                {
                    double penaltyMs = (cycleTimeMs < ueOnStart) ? (ueOnStart - cycleTimeMs) : ((m_cycleMs - cycleTimeMs) + ueOnStart);
                    rawDelayMs += penaltyMs;
                }
            }

            m_delaysMs.push_back(rawDelayMs);
        }
        m_totalRxBytes += packet->GetSize();
    }

    void RecordDrop(Ptr<const Packet> packet)
    {
        m_totalDropBytes += packet->GetSize();
    }

    uint64_t GetPendingBytes() const
    {
        uint64_t processedBytes = m_totalRxBytes + m_totalDropBytes;
        return (m_totalTxBytes >= processedBytes) ? (m_totalTxBytes - processedBytes) : 0;
    }

    double GetP99Latency() const
    {
        if (m_delaysMs.empty()) return 0.0;
        std::vector<double> sortedDelays = m_delaysMs;
        std::sort(sortedDelays.begin(), sortedDelays.end());
        size_t idxP99 = static_cast<size_t>(0.99 * sortedDelays.size());
        if (idxP99 >= sortedDelays.size()) idxP99 = sortedDelays.size() - 1;
        return sortedDelays[idxP99];
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
    bool m_enableCdrx;
    bool m_enableDciDynamic;
    double m_phaseOffsetMs;
    double m_cdrxOnDurationMs;
    double m_dynamicOnDurationMs;
    double m_cycleMs;
    std::unordered_map<uint64_t, Time> m_txTimes;
    std::vector<double> m_delaysMs;
    uint64_t m_totalTxBytes;
    uint64_t m_totalRxBytes;
    uint64_t m_totalDropBytes;
};

// 宣告 UE 追蹤器前置介面
class UeEnergyTracker;

// ----------------------------------------------------------------------------------
// 3. 閉迴路動態佇列感知控制器 (支援時域 Active Window + 頻域 Dynamic BWP 2D 對齊)
// ----------------------------------------------------------------------------------
class DynamicDtxController
{
public:
    DynamicDtxController(PacketDelayTracker* tracker, UeEnergyTracker* ueTracker, ApplicationContainer apps,
                         double totalDataRateMbps, bool enableDciDynamic = false, bool enableBwpDynamic = false,
                         double linkDelaySec = 0.010, double evalIntervalMs = 40.0, double minActiveMs = 10.0,
                         double maxActiveMs = 30.0, double stepUpMs = 4.0, double stepDownMs = 2.0,
                         uint64_t backlogThresholdBytes = 2000,
                         std::string timelineCsvName = "active_window_timeline.csv")
        : m_tracker(tracker),
          m_ueTracker(ueTracker),
          m_apps(apps),
          m_totalDataRateMbps(totalDataRateMbps),
          m_enableDciDynamic(enableDciDynamic),
          m_enableBwpDynamic(enableBwpDynamic),
          m_evalIntervalSec(evalIntervalMs / 1000.0),
          m_minActiveMs(minActiveMs),
          m_maxActiveMs(maxActiveMs),
          m_stepUpMs(stepUpMs),
          m_stepDownMs(stepDownMs),
          m_currentActiveMs(minActiveMs),
          m_currentBwp(BWP_NARROW_20MHZ),
          m_bwpInactivityTimerMs(0.0),
          m_bwpInactivityTimeoutMs(10.0),
          m_cellDtxPeriodMs(40.0),
          m_avgTbsPerSlot(3000),
          m_slotDurationMs(0.5),
          m_backlogThresholdBytes(backlogThresholdBytes),
          m_timelineCsvName(timelineCsvName)
    {
        m_bdpThresholdBytes = static_cast<uint64_t>((totalDataRateMbps * 1e6 * linkDelaySec) / 8.0);
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

        Simulator::Schedule(Seconds(m_evalIntervalSec), &DynamicDtxController::EvaluateAndAdjust, this);
    }

    void EvaluateAndAdjust();

    double GetCurrentActiveMs() const { return m_currentActiveMs; }
    BwpState GetCurrentBwp() const { return m_currentBwp; }

private:
    PacketDelayTracker* m_tracker;
    UeEnergyTracker* m_ueTracker;
    ApplicationContainer m_apps;
    double m_totalDataRateMbps;
    bool m_enableDciDynamic;
    bool m_enableBwpDynamic;
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
};

// ----------------------------------------------------------------------------------
// 4. 基地台功耗追蹤器 (gNB Energy Tracker)
// ----------------------------------------------------------------------------------
class GnbSlotEnergyTracker
{
public:
    GnbSlotEnergyTracker(double slotDurationMs, double p0Active, double pMicroSleep, double pTxMax, double deltaP)
        : m_slotDurationSec(slotDurationMs / 1000.0),
          m_p0Active(p0Active),
          m_pMicroSleep(pMicroSleep),
          m_pTxMax(pTxMax),
          m_deltaP(deltaP),
          m_totalEnergyJoules(0.0),
          m_activeSlots(0),
          m_sleepSlots(0),
          m_lastTotalRxBytes(0)
    {}

    void TrackSlot(Ptr<FlowMonitor> monitor, double maxCapacityMbps)
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
            double slotLoad = std::min(1.0, slotThroughputMbps / maxCapacityMbps);
            slotPower = m_p0Active + (m_deltaP * m_pTxMax * slotLoad);
        }
        else
        {
            m_sleepSlots++;
            slotPower = m_pMicroSleep;
        }

        m_totalEnergyJoules += slotPower * m_slotDurationSec;
        Simulator::Schedule(Seconds(m_slotDurationSec), &GnbSlotEnergyTracker::TrackSlot, this, monitor, maxCapacityMbps);
    }

    double GetAvgPower() const { return (m_activeSlots + m_sleepSlots > 0) ? m_totalEnergyJoules / ((m_activeSlots + m_sleepSlots) * m_slotDurationSec) : 0.0; }
    double GetTotalEnergy() const { return m_totalEnergyJoules; }

private:
    double m_slotDurationSec;
    double m_p0Active;
    double m_pMicroSleep;
    double m_pTxMax;
    double m_deltaP;
    double m_totalEnergyJoules;
    uint32_t m_activeSlots;
    uint32_t m_sleepSlots;
    uint64_t m_lastTotalRxBytes;
};

// ----------------------------------------------------------------------------------
// 5. UE 手機端功耗追蹤器 (支援 DCI 級別動態時域對齊 ＋ 3GPP BWP 2D 頻域動態切換)
// ----------------------------------------------------------------------------------
class UeEnergyTracker
{
public:
    UeEnergyTracker(uint32_t numUes, double slotDurationMs, bool enableCdrx, bool enableDciDynamic = false,
                    bool enableBwpDynamic = false, double phaseOffsetMs = 0.0, double cdrxOnDurationMs = 10.0,
                    double pDeepSleepMw = 1.5)
        : m_numUes(numUes),
          m_slotDurationSec(slotDurationMs / 1000.0),
          m_enableCdrx(enableCdrx),
          m_enableDciDynamic(enableDciDynamic),
          m_enableBwpDynamic(enableBwpDynamic),
          m_phaseOffsetMs(phaseOffsetMs),
          m_cdrxOnDurationMs(cdrxOnDurationMs),
          m_dynamicOnDurationMs(cdrxOnDurationMs),
          m_currentBwp(BWP_NARROW_20MHZ),
          m_pDeepSleepMw(pDeepSleepMw),
          m_totalEnergyJoules(0.0),
          m_switchingDelayRemainingMs(0.0),
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
            // 3GPP TS 38.133 BWP Switching Delay Penalty (1.0 ms Interruption Gap)
            m_switchingDelayRemainingMs = 1.0;
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

        // 處理 BWP 切換中斷延遲懲罰
        if (m_switchingDelayRemainingMs > 0.0)
        {
            m_switchingDelayRemainingMs -= (m_slotDurationSec * 1000.0);
            if (m_switchingDelayRemainingMs < 0.0) m_switchingDelayRemainingMs = 0.0;
        }

        // 取得當前 Active BWP 之 3GPP 實體層功耗 (20MHz vs 100MHz)
        double currentRxPowerMw = m_enableBwpDynamic ? g_bwpPowerProfiles[m_currentBwp].pRxMw : g_bwpPowerProfiles[BWP_WIDE_100MHZ].pRxMw;
        double currentMicroSleepPowerMw = m_enableBwpDynamic ? g_bwpPowerProfiles[m_currentBwp].pMicroSleepMw : g_bwpPowerProfiles[BWP_WIDE_100MHZ].pMicroSleepMw;

        for (uint32_t i = 0; i < m_numUes; ++i)
        {
            uint64_t bytesInSlot = (currentRxBytes[i] >= m_lastRxBytesPerUe[i]) ? (currentRxBytes[i] - m_lastRxBytesPerUe[i]) : 0;
            m_lastRxBytesPerUe[i] = currentRxBytes[i];

            double uePowerMw = 0.0;
            if (bytesInSlot > 0 && m_switchingDelayRemainingMs <= 0.0)
            {
                uePowerMw = currentRxPowerMw;
            }
            else if (!m_enableCdrx)
            {
                uePowerMw = currentMicroSleepPowerMw;
            }
            else
            {
                if (inUeOnDuration)
                {
                    uePowerMw = currentMicroSleepPowerMw;
                }
                else
                {
                    uePowerMw = m_pDeepSleepMw;
                }
            }

            double energyJoules = (uePowerMw / 1000.0) * m_slotDurationSec;
            m_ueEnergyJoules[i] += energyJoules;
            m_totalEnergyJoules += energyJoules;
        }

        Simulator::Schedule(Seconds(m_slotDurationSec), &UeEnergyTracker::TrackSlot, this, monitor, controller);
    }

    double GetAvgPowerMwPerUe() const
    {
        double totalTime = Simulator::Now().GetSeconds();
        if (totalTime <= 0 || m_numUes == 0) return 0.0;
        return (m_totalEnergyJoules / totalTime / m_numUes) * 1000.0;
    }

    double GetTotalEnergyJoules() const { return m_totalEnergyJoules; }

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
    double m_switchingDelayRemainingMs;
    std::vector<uint64_t> m_lastRxBytesPerUe;
    std::vector<double> m_ueEnergyJoules;
};

// DynamicDtxController 函數成員實作 (包含時域 On-Duration ＋ 頻域 BWP 2D 雙重動態調適)
void DynamicDtxController::EvaluateAndAdjust()
{
    uint64_t totalPending = m_tracker ? m_tracker->GetPendingBytes() : 0;
    uint64_t rlcBacklogBytes = (totalPending > m_bdpThresholdBytes) ? (totalPending - m_bdpThresholdBytes) : 0;

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

        // 🛠️ 2D 頻域動態：佇列積壓時，即時升頻至 100 MHz 寬頻 BWP 全力衝刺，並重設 Inactivity Timer
        if (m_enableBwpDynamic)
        {
            m_currentBwp = BWP_WIDE_100MHZ;
            m_bwpInactivityTimerMs = m_bwpInactivityTimeoutMs;
        }
    }
    else
    {
        m_currentActiveMs = std::max(m_minActiveMs, m_currentActiveMs - m_stepDownMs);

        // 🛠️ 2D 頻域動態：佇列清空時，倒數 Inactivity Timer，歸零後切回 20 MHz 窄頻 BWP 極致省電
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

    // 觸發時域與頻域之屬性同步更新
    if (std::abs(m_currentActiveMs - oldActiveMs) > 1e-3 || m_currentBwp != oldBwp)
    {
        for (uint32_t i = 0; i < m_apps.GetN(); ++i)
        {
            Ptr<Application> app = m_apps.Get(i);
            app->SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(m_currentActiveMs / 1000.0) + "]"));
            app->SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string((m_cellDtxPeriodMs - m_currentActiveMs) / 1000.0) + "]"));
        }

        // 🛠️ 觸發 DCI format 2_6 廣播，將動態 Active Window 與 BWP 狀態同步至 UE
        if (m_enableDciDynamic && m_tracker)
        {
            m_tracker->UpdateDynamicOnDurationMs(m_currentActiveMs);
        }

        if (m_ueTracker)
        {
            if (m_enableDciDynamic) m_ueTracker->UpdateDynamicOnDurationMs(m_currentActiveMs);
            if (m_enableBwpDynamic) m_ueTracker->UpdateBwpState(m_currentBwp);
        }

        NS_LOG_INFO("[Closed-Loop DTX + DCI + BWP 2D] Active Window: " << m_currentActiveMs << " ms, BWP: " << (m_currentBwp == BWP_WIDE_100MHZ ? "100MHz" : "20MHz"));
    }

    std::ofstream csv(m_timelineCsvName, std::ios::app);
    if (csv.is_open())
    {
        csv << Simulator::Now().GetSeconds() << ","
            << m_currentActiveMs << ","
            << (m_currentBwp == BWP_WIDE_100MHZ ? 100 : 20) << ","
            << totalPending << ","
            << rlcBacklogBytes << ","
            << m_totalDataRateMbps << "\n";
        csv.close();
    }

    Simulator::Schedule(Seconds(m_evalIntervalSec), &DynamicDtxController::EvaluateAndAdjust, this);
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
    uint32_t numUes = 3;
    double simTime = 5.0;
    double udpPacketRateMbps = 2.0;
    bool enableGating = true;
    bool enableQueueAware = false;
    bool enableCdrx = false;
    bool enableDciDynamic = false;
    bool enableBwpDynamic = false;
    double phaseOffsetMs = 0.0;
    uint32_t cdrxOnDurationMs = 10;
    uint32_t cdrxInactivityMs = 10;
    double cellDtxPeriodMs = 40.0;
    double cellDtxActiveMs = 10.0;
    double qosTargetDelayMs = 20.0;
    std::string csvFileName = "joint_energy_results.csv";

    CommandLine cmd(__FILE__);
    cmd.AddValue("numUes", "Number of UEs", numUes);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("udpDataRate", "UDP Downlink Data Rate in Mbps per UE", udpPacketRateMbps);
    cmd.AddValue("enableGating", "Enable Cell-DTX Scheduler Gating", enableGating);
    cmd.AddValue("enableQueueAware", "Enable Closed-Loop Dynamic Controller", enableQueueAware);
    cmd.AddValue("enableCdrx", "Enable UE C-DRX Energy Saving Mode", enableCdrx);
    cmd.AddValue("enableDciDynamic", "Enable 3GPP DCI-Level Dynamic Alignment", enableDciDynamic);
    cmd.AddValue("enableBwpDynamic", "Enable DCI-based Dynamic BWP Switching (20MHz <-> 100MHz)", enableBwpDynamic);
    cmd.AddValue("phaseOffsetMs", "Phase Offset between gNB and UE C-DRX in ms", phaseOffsetMs);
    cmd.AddValue("cdrxOnDurationMs", "C-DRX OnDuration Timer in ms", cdrxOnDurationMs);
    cmd.AddValue("cdrxInactivityMs", "C-DRX Inactivity Timer in ms", cdrxInactivityMs);
    cmd.AddValue("cellDtxPeriodMs", "Cell-DTX Period in ms", cellDtxPeriodMs);
    cmd.AddValue("cellDtxActiveMs", "Cell-DTX Active Duration in ms", cellDtxActiveMs);
    cmd.AddValue("qosTargetDelayMs", "QoS Target Delay Bound in ms", qosTargetDelayMs);
    cmd.AddValue("csvFileName", "Output CSV file name", csvFileName);
    cmd.Parse(argc, argv);

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
    uePositionAlloc->Add(Vector(30.0, 0.0, 1.5));
    uePositionAlloc->Add(Vector(60.0, 0.0, 1.5));
    uePositionAlloc->Add(Vector(150.0, 0.0, 1.5));
    mobility.SetPositionAllocator(uePositionAlloc);
    mobility.Install(ueNodes);

    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(3.5e9, 100e6, 1);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    channelHelper->AssignChannelsToBands({band});

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);

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

    PacketDelayTracker delayTracker(enableCdrx, enableDciDynamic, phaseOffsetMs, cdrxOnDurationMs, cellDtxPeriodMs);
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

        OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(ueIpIface.GetAddress(i), port + i));
        onoff.SetAttribute("PacketSize", UintegerValue(1024));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate(std::to_string(udpPacketRateMbps) + "Mbps")));

        if (enableGating)
        {
            onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(cellDtxActiveMs / 1000.0) + "]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string((cellDtxPeriodMs - cellDtxActiveMs) / 1000.0) + "]"));
        }
        else
        {
            onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=100.0]"));
            onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
        }

        clientApps.Add(onoff.Install(remoteHost));
    }

    serverApps.Start(Seconds(0.01));
    clientApps.Start(Seconds(0.05));

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    GnbSlotEnergyTracker gnbSlotTracker(1.0, 150.0, 15.0, 40.0, 4.7);
    Simulator::Schedule(MilliSeconds(1), &GnbSlotEnergyTracker::TrackSlot, &gnbSlotTracker, monitor, 100.0);

    UeEnergyTracker ueSlotTracker(numUes, 1.0, enableCdrx, enableDciDynamic, enableBwpDynamic, phaseOffsetMs, cdrxOnDurationMs, 1.5);

    double totalTrafficRateMbps = udpPacketRateMbps * numUes;
    DynamicDtxController dynamicController(&delayTracker, &ueSlotTracker, clientApps, totalTrafficRateMbps, enableDciDynamic, enableBwpDynamic, 0.010, 40.0, 10.0, 30.0, 4.0, 2.0, 2000, "active_window_timeline.csv");
    if (enableGating && enableQueueAware)
    {
        dynamicController.Start();
    }

    Simulator::Schedule(MilliSeconds(1), &UeEnergyTracker::TrackSlot, &ueSlotTracker, monitor, &dynamicController);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer flowStats = monitor->GetFlowStats();
    uint64_t totalRxBytes = 0;
    for (auto iter = flowStats.begin(); iter != flowStats.end(); ++iter)
    {
        totalRxBytes += iter->second.rxBytes;
    }

    double finalActiveMs = enableQueueAware ? dynamicController.GetCurrentActiveMs() : cellDtxActiveMs;
    double avgPower = gnbSlotTracker.GetAvgPower();
    double totalEnergy = gnbSlotTracker.GetTotalEnergy();
    double totalRxMbits = (totalRxBytes * 8.0) / 1e6;
    double energyEff = (totalEnergy > 0) ? (totalRxMbits / totalEnergy) : 0.0;
    double p99Latency = delayTracker.GetP99Latency();
    double qosSatisfaction = delayTracker.GetQosSatisfactionRate(qosTargetDelayMs);

    double ueAvgPowerMw = ueSlotTracker.GetAvgPowerMwPerUe();
    double ueTotalEnergy = ueSlotTracker.GetTotalEnergyJoules();
    double ueBatteryMah = ueSlotTracker.GetBatteryConsumptionMah();
    double ue3EnergyJoules = ueSlotTracker.GetUeEnergyJoules(2);

    ExportStatsToCsv(csvFileName, simTime, enableGating, enableQueueAware, cellDtxPeriodMs, finalActiveMs, udpPacketRateMbps, totalRxBytes, avgPower, totalEnergy, energyEff, p99Latency, qosSatisfaction, enableCdrx, enableDciDynamic, enableBwpDynamic, phaseOffsetMs, ueAvgPowerMw, ueTotalEnergy, ueBatteryMah, ue3EnergyJoules);

    Simulator::Destroy();
    return 0;
}
