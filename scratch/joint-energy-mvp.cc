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
#include <map>
#include <cmath>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

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

class BwpHandoverProbe
{
public:
    BwpHandoverProbe(const std::vector<Ptr<NrMacSchedulerNs3>>& schedulers,
                     const std::vector<Ptr<NrGnbPhy>>& phys,
                     std::string csvFileName,
                     double sampleIntervalMs)
        : m_schedulers(schedulers),
          m_csvFileName(std::move(csvFileName)),
          m_sampleInterval(Seconds(sampleIntervalMs / 1000.0)),
          m_dlScheduledUes(schedulers.size(), 0),
          m_dlUsedSymbols(schedulers.size(), 0),
          m_dlUsedRegs(schedulers.size(), 0)
    {
        NS_ABORT_MSG_IF(m_schedulers.size() != phys.size() || m_schedulers.empty(),
                        "BWP handover probe requires matching scheduler/PHY vectors");
        NS_ABORT_MSG_IF(sampleIntervalMs <= 0.0,
                        "BWP handover probe sample interval must be positive");
        for (const auto& phy : phys)
        {
            NS_ABORT_MSG_IF(!phy, "BWP handover probe received a null PHY");
            phy->TraceConnectWithoutContext(
                "SlotDataStats",
                MakeCallback(&BwpHandoverProbe::OnSlotData, this));
        }
    }

    void Start(uint8_t initialBwpId)
    {
        m_expectedActiveBwp = initialBwpId;
        m_csv.open(m_csvFileName, std::ios::trunc);
        NS_ABORT_MSG_IF(!m_csv.is_open(),
                        "Unable to open BWP handover trace " << m_csvFileName);
        m_csv << "SimTimeSec,ExpectedActiveBwp,Bwp0BacklogBytes,Bwp1BacklogBytes,"
                 "Bwp0DlScheduledUes,Bwp1DlScheduledUes,Bwp0DlUsedSymbols,"
                 "Bwp1DlUsedSymbols,Bwp0DlUsedRegs,Bwp1DlUsedRegs\n";
        Simulator::ScheduleNow(&BwpHandoverProbe::Sample, this);
    }

    void SetExpectedActiveBwp(uint8_t bwpId)
    {
        NS_ABORT_MSG_IF(bwpId >= m_schedulers.size(),
                        "BWP handover probe target is out of range");
        m_expectedActiveBwp = bwpId;
    }

    void OnSlotData(const SfnSf&,
                    uint32_t scheduledUe,
                    uint32_t usedReg,
                    uint32_t usedSym,
                    uint32_t,
                    uint32_t,
                    uint16_t bwpId,
                    uint16_t)
    {
        NS_ABORT_MSG_IF(bwpId >= m_schedulers.size(),
                        "SlotDataStats reported an unexpected BWP ID");
        m_dlScheduledUes[bwpId] += scheduledUe;
        m_dlUsedSymbols[bwpId] += usedSym;
        m_dlUsedRegs[bwpId] += usedReg;
    }

private:
    void Sample()
    {
        NS_ABORT_MSG_IF(m_schedulers.size() < 2,
                        "The handover trace currently expects BWP0 and BWP1");
        m_csv << std::fixed << std::setprecision(9)
              << Simulator::Now().GetSeconds() << ","
              << +m_expectedActiveBwp << ","
              << m_schedulers[0]->GetTotalDlBufferBytes() << ","
              << m_schedulers[1]->GetTotalDlBufferBytes() << ","
              << m_dlScheduledUes[0] << ","
              << m_dlScheduledUes[1] << ","
              << m_dlUsedSymbols[0] << ","
              << m_dlUsedSymbols[1] << ","
              << m_dlUsedRegs[0] << ","
              << m_dlUsedRegs[1] << "\n";
        std::fill(m_dlScheduledUes.begin(), m_dlScheduledUes.end(), 0);
        std::fill(m_dlUsedSymbols.begin(), m_dlUsedSymbols.end(), 0);
        std::fill(m_dlUsedRegs.begin(), m_dlUsedRegs.end(), 0);
        Simulator::Schedule(m_sampleInterval, &BwpHandoverProbe::Sample, this);
    }

    std::vector<Ptr<NrMacSchedulerNs3>> m_schedulers;
    std::string m_csvFileName;
    Time m_sampleInterval;
    uint8_t m_expectedActiveBwp{BWP_NARROW_20MHZ};
    std::vector<uint64_t> m_dlScheduledUes;
    std::vector<uint64_t> m_dlUsedSymbols;
    std::vector<uint64_t> m_dlUsedRegs;
    std::ofstream m_csv;
};

class TbContinuityProbe
{
public:
    TbContinuityProbe(const NetDeviceContainer& gnbDevices,
                      const NetDeviceContainer& ueDevices,
                      uint32_t bwpCount,
                      std::string csvFileName)
        : m_csvFileName(std::move(csvFileName))
    {
        NS_ABORT_MSG_IF(gnbDevices.GetN() != 1 || bwpCount != 2,
                        "TB continuity probe currently expects one gNB and two BWPs");

        Ptr<NrGnbPhy> bwp0Phy = NrHelper::GetGnbPhy(gnbDevices.Get(0), BWP_NARROW_20MHZ);
        Ptr<NrGnbPhy> bwp1Phy = NrHelper::GetGnbPhy(gnbDevices.Get(0), BWP_WIDE_100MHZ);
        NS_ABORT_MSG_IF(!bwp0Phy || !bwp1Phy, "TB continuity probe could not access gNB PHYs");
        bwp0Phy->GetSpectrumPhy()->TraceConnectWithoutContext(
            "TxPacketTraceGnb",
            MakeCallback(&TbContinuityProbe::OnGnbTxBwp0, this));
        bwp1Phy->GetSpectrumPhy()->TraceConnectWithoutContext(
            "TxPacketTraceGnb",
            MakeCallback(&TbContinuityProbe::OnGnbTxBwp1, this));

        for (uint32_t ueId = 0; ueId < ueDevices.GetN(); ++ueId)
        {
            for (uint32_t bwpId = 0; bwpId < bwpCount; ++bwpId)
            {
                Ptr<NrUePhy> uePhy = NrHelper::GetUePhy(ueDevices.Get(ueId), bwpId);
                NS_ABORT_MSG_IF(!uePhy, "TB continuity probe could not access UE PHY");
                uePhy->GetSpectrumPhy()->TraceConnectWithoutContext(
                    "RxPacketTraceUe",
                    MakeCallback(&TbContinuityProbe::OnUeRx, this));
            }
        }

        for (uint32_t bwpId = 0; bwpId < bwpCount; ++bwpId)
        {
            Ptr<NrGnbMac> mac = NrHelper::GetGnbMac(gnbDevices.Get(0), bwpId);
            NS_ABORT_MSG_IF(!mac, "TB continuity probe could not access gNB MAC");
            mac->TraceConnectWithoutContext(
                "DlHarqFeedback",
                MakeCallback(&TbContinuityProbe::OnDlHarqFeedback, this));
        }
    }

    void Start(uint8_t initialBwpId)
    {
        m_expectedActiveBwp = initialBwpId;
        m_csv.open(m_csvFileName, std::ios::trunc);
        NS_ABORT_MSG_IF(!m_csv.is_open(),
                        "Unable to open TB continuity trace " << m_csvFileName);
        m_csv << "SimTimeSec,Event,ExpectedActiveBwp,ObservedBwp,Rnti,HarqProcessId,"
                 "Rv,TbSizeBytes,Corrupt,HarqStatus,NumRetx\n";
    }

    void SetExpectedActiveBwp(uint8_t bwpId)
    {
        NS_ABORT_MSG_IF(bwpId > BWP_WIDE_100MHZ,
                        "TB continuity probe target is out of range");
        m_expectedActiveBwp = bwpId;
    }

    void OnGnbTxBwp0(GnbPhyPacketCountParameter params)
    {
        WriteTx(BWP_NARROW_20MHZ, params);
    }

    void OnGnbTxBwp1(GnbPhyPacketCountParameter params)
    {
        WriteTx(BWP_WIDE_100MHZ, params);
    }

    void OnUeRx(RxPacketTraceParams params)
    {
        m_csv << std::fixed << std::setprecision(9)
              << Simulator::Now().GetSeconds() << ",RX,"
              << +m_expectedActiveBwp << ","
              << params.m_bwpId << ","
              << params.m_rnti << ","
              << +params.m_harqProcessId << ","
              << +params.m_rv << ","
              << params.m_tbSize << ","
              << (params.m_corrupt ? 1 : 0) << ",NA,0\n";
    }

    void OnDlHarqFeedback(const DlHarqInfo& info)
    {
        m_csv << std::fixed << std::setprecision(9)
              << Simulator::Now().GetSeconds() << ",HARQ,"
              << +m_expectedActiveBwp << ","
              << +info.m_bwpIndex << ","
              << info.m_rnti << ","
              << +info.m_harqProcessId << ",0,0,0,"
              << (info.m_harqStatus == DlHarqInfo::ACK ? "ACK" : "NACK") << ","
              << +info.m_numRetx << "\n";
    }

private:
    void WriteTx(uint8_t bwpId, const GnbPhyPacketCountParameter& params)
    {
        m_csv << std::fixed << std::setprecision(9)
              << Simulator::Now().GetSeconds() << ",TX,"
              << +m_expectedActiveBwp << ","
              << +bwpId << ",0,0,0,"
              << params.m_noBytes << ",0,NA,0\n";
    }

    std::string m_csvFileName;
    uint8_t m_expectedActiveBwp{BWP_NARROW_20MHZ};
    std::ofstream m_csv;
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
    std::cout << "PhysicalBwpSwitchTimeSec=" << Simulator::Now().GetSeconds()
              << ", TargetBwpId=" << +targetBwpId
              << ", TargetBwpMHz="
              << g_bwpResourceProfiles[targetBwpId].channelBwMhz << std::endl;
}

static void
SwitchUeToPrimaryBwp(NetDeviceContainer ueDevices, uint32_t ueId, uint8_t targetBwpId)
{
    NS_ABORT_MSG_IF(ueId >= ueDevices.GetN(), "Per-UE BWP index is out of range");
    Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevices.Get(ueId));
    NS_ABORT_MSG_IF(!ueDevice || !ueDevice->GetRrc(),
                    "Unable to access UE RRC for physical BWP switch");
    NS_ABORT_MSG_IF(!ueDevice->GetRrc()->RequestPrimaryBwpSwitch(targetBwpId),
                    "UE " << ueId << " rejected primary BWP switch to " << +targetBwpId);
    std::cout << "PhysicalPerUeBwpSwitchTimeSec=" << Simulator::Now().GetSeconds()
              << ", UeId=" << ueId << ", TargetBwpId=" << +targetBwpId
              << ", TargetBwpMHz=" << g_bwpResourceProfiles[targetBwpId].channelBwMhz
              << std::endl;
}

static void
ApplyUeEnergyBwpScaling(energy::DeviceEnergyModelContainer ueEnergyModels,
                        uint32_t bandwidthMhz)
{
    for (uint32_t i = 0; i < ueEnergyModels.GetN(); ++i)
    {
        Ptr<NrUeEnergyModel> model =
            DynamicCast<NrUeEnergyModel>(ueEnergyModels.Get(i));
        NS_ABORT_MSG_IF(!model, "Expected an NrUeEnergyModel for UE " << i);
        model->ApplyBwpScaling(bandwidthMhz);
    }
}

static void
ApplyPerUeEnergyBwpScaling(energy::DeviceEnergyModelContainer ueEnergyModels,
                           std::vector<uint32_t> perUeBwpMHz)
{
    NS_ABORT_MSG_IF(ueEnergyModels.GetN() != perUeBwpMHz.size(),
                    "Per-UE BWP energy list does not match UE energy models");
    for (uint32_t i = 0; i < ueEnergyModels.GetN(); ++i)
    {
        Ptr<NrUeEnergyModel> model = DynamicCast<NrUeEnergyModel>(ueEnergyModels.Get(i));
        NS_ABORT_MSG_IF(!model, "Expected an NrUeEnergyModel for UE " << i);
        model->ApplyBwpScaling(perUeBwpMHz[i]);
    }
}

static void
SwitchUeToPrimaryBwpAndEnergy(NetDeviceContainer ueDevices,
                              energy::DeviceEnergyModelContainer ueEnergyModels,
                              uint32_t ueId,
                              uint8_t targetBwpId)
{
    SwitchUeToPrimaryBwp(ueDevices, ueId, targetBwpId);
    NS_ABORT_MSG_IF(ueId >= ueEnergyModels.GetN(),
                    "Per-UE BWP energy index is out of range");
    Ptr<NrUeEnergyModel> model = DynamicCast<NrUeEnergyModel>(ueEnergyModels.Get(ueId));
    NS_ABORT_MSG_IF(!model, "Expected an NrUeEnergyModel for UE " << ueId);
    model->ApplyBwpScaling(g_bwpResourceProfiles[targetBwpId].channelBwMhz);
}

static void
SwitchAllUesToPrimaryBwpAndEnergy(
    NetDeviceContainer ueDevices,
    energy::DeviceEnergyModelContainer ueEnergyModels,
    uint8_t targetBwpId)
{
    SwitchAllUesToPrimaryBwp(ueDevices, targetBwpId);
    ApplyUeEnergyBwpScaling(
        ueEnergyModels,
        static_cast<uint32_t>(g_bwpResourceProfiles[targetBwpId].channelBwMhz));
}

static void
SwitchAllUesToPrimaryBwpAndEnergyWithProbe(
    NetDeviceContainer ueDevices,
    energy::DeviceEnergyModelContainer ueEnergyModels,
    BwpHandoverProbe* probe,
    TbContinuityProbe* tbProbe,
    uint8_t targetBwpId)
{
    SwitchAllUesToPrimaryBwpAndEnergy(ueDevices, ueEnergyModels, targetBwpId);
    if (probe)
    {
        probe->SetExpectedActiveBwp(targetBwpId);
    }
    if (tbProbe)
    {
        tbProbe->SetExpectedActiveBwp(targetBwpId);
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

    int StepIpc(double backlogBytes, double maxUeBacklogBytes,
                double totalRlcOccupancyRatio, double maxBearerOccupancyRatio,
                double offeredRateMbps,
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
                      "S4,%.2f,%.2f,%.6f,%.6f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.4f\n",
                      backlogBytes, maxUeBacklogBytes, totalRlcOccupancyRatio,
                      maxBearerOccupancyRatio, offeredRateMbps, deliveredRateMbps, delayMs,
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

    bool StepIpcS5(double backlogBytes,
                   double maxUeBacklogBytes,
                   double totalRlcOccupancyRatio,
                   double maxBearerOccupancyRatio,
                   double offeredRateMbps,
                   double deliveredRateMbps,
                   double delayMs,
                   double avgCqi,
                   double p10Cqi,
                   double minCqi,
                   double avgMcs,
                   double wActiveMs,
                   const std::vector<BwpState>& perUeBwp,
                   const std::vector<uint32_t>& perUeBufferBytes,
                   const std::vector<double>& perUeDelayMs,
                   double gnbPowerW,
                   double uePowerMw,
                   double jainFairness,
                   int& activeActionIndex,
                   std::vector<BwpState>& nextPerUeBwp)
    {
        if (!m_connected && !Connect()) return false;
        std::ostringstream wire;
        wire << "S5," << std::fixed << std::setprecision(2)
             << backlogBytes << "," << maxUeBacklogBytes << ","
             << std::setprecision(6) << totalRlcOccupancyRatio << ","
             << maxBearerOccupancyRatio << ","
             << std::setprecision(4) << offeredRateMbps << "," << deliveredRateMbps << ","
             << std::setprecision(2) << delayMs << "," << avgCqi << "," << p10Cqi << ","
             << minCqi << "," << avgMcs << "," << wActiveMs << "," << gnbPowerW << ","
             << uePowerMw << "," << std::setprecision(4) << jainFairness;
        for (auto bwp : perUeBwp) wire << "," << static_cast<int>(bwp);
        for (auto value : perUeBufferBytes) wire << "," << value;
        for (auto value : perUeDelayMs) wire << "," << std::setprecision(2) << value;
        wire << "\n";
        const std::string message = wire.str();
        if (send(m_sockFd, message.c_str(), message.size(), 0) < 0)
        {
            m_connected = false;
            return false;
        }
        char recvBuffer[256] = {};
        ssize_t bytesRead = recv(m_sockFd, recvBuffer, sizeof(recvBuffer) - 1, 0);
        if (bytesRead <= 0)
        {
            m_connected = false;
            return false;
        }
        std::stringstream response(std::string(recvBuffer, bytesRead));
        std::string token;
        if (!std::getline(response, token, ',')) return false;
        try { activeActionIndex = std::stoi(token); }
        catch (...) { return false; }
        nextPerUeBwp.clear();
        while (std::getline(response, token, ','))
        {
            try
            {
                const int value = std::stoi(token);
                if (value != 0 && value != 1) return false;
                nextPerUeBwp.push_back(static_cast<BwpState>(value));
            }
            catch (...) { return false; }
        }
        return nextPerUeBwp.size() == perUeBwp.size();
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
    explicit PacketDelayTracker(uint32_t numUes)
        : m_delaysMsPerUe(numUes),
          m_windowDelaysMsPerUe(numUes),
          m_totalTxBytes(0),
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

    void RecordRxForUe(uint32_t ueId,
                       Ptr<const Packet> packet,
                       const Address& from,
                       const Address& to)
    {
        NS_ABORT_MSG_IF(ueId >= m_delaysMsPerUe.size(), "RX trace has an invalid UE index");
        uint64_t uid = packet->GetUid();
        auto it = m_txInfo.find(uid);
        if (it != m_txInfo.end())
        {
            Time sendTime = it->second.first;
            Time delay = Simulator::Now() - sendTime;
            double rawDelayMs = delay.GetSeconds() * 1000.0;
            m_delaysMs.push_back(rawDelayMs);
            m_delaysMsPerUe[ueId].push_back(rawDelayMs);
            m_windowDelaysMsPerUe[ueId].push_back(rawDelayMs);
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

    void RecordDropWithContext(std::string context, Ptr<const Packet> packet)
    {
        RecordDrop(packet);
        const std::string ueMarker = "/UeMap/";
        const std::string bearerMarker = "/DataRadioBearerMap/";
        const size_t uePos = context.find(ueMarker);
        const size_t bearerPos = context.find(bearerMarker);
        if (uePos == std::string::npos || bearerPos == std::string::npos)
        {
            return;
        }
        const size_t ueBegin = uePos + ueMarker.size();
        const size_t ueEnd = context.find('/', ueBegin);
        const size_t lcidBegin = bearerPos + bearerMarker.size();
        const size_t lcidEnd = context.find('/', lcidBegin);
        if (ueEnd == std::string::npos || lcidEnd == std::string::npos)
        {
            return;
        }
        const uint16_t rnti = static_cast<uint16_t>(std::stoul(context.substr(ueBegin, ueEnd - ueBegin)));
        const uint8_t lcid = static_cast<uint8_t>(std::stoul(context.substr(lcidBegin, lcidEnd - lcidBegin)));
        auto& counters = m_dropByBearer[{rnti, lcid}];
        counters.first++;
        counters.second += packet->GetSize();
    }

    const std::map<std::pair<uint16_t, uint8_t>, std::pair<uint64_t, uint64_t>>&
    GetDropByBearer() const
    {
        return m_dropByBearer;
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

    uint64_t GetTotalDropBytes() const
    {
        return m_totalDropBytes;
    }

    double GetP99Latency() const
    {
        return CalculatePercentile(m_delaysMs, 0.99);
    }

    double GetLatencyPercentile(double percentile) const
    {
        return CalculatePercentile(m_delaysMs, percentile);
    }

    double GetWindowP99Latency() const
    {
        return CalculatePercentile(m_windowDelaysMs, 0.99);
    }

    std::vector<double> GetWindowP99LatencyPerUe() const
    {
        std::vector<double> values;
        values.reserve(m_windowDelaysMsPerUe.size());
        for (const auto& delays : m_windowDelaysMsPerUe)
        {
            values.push_back(CalculatePercentile(delays, 0.99));
        }
        return values;
    }

    void ResetWindowStats()
    {
        m_windowDelaysMs.clear();
        for (auto& delays : m_windowDelaysMsPerUe) delays.clear();
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
        return CalculateQosSatisfactionRate(m_delaysMs, targetDelayMs);
    }

    std::string GetPerUeLatencyPercentile(double percentile) const
    {
        std::ostringstream values;
        values << std::fixed << std::setprecision(4);
        for (uint32_t ueId = 0; ueId < m_delaysMsPerUe.size(); ++ueId)
        {
            if (ueId > 0)
            {
                values << ";";
            }
            values << CalculatePercentile(m_delaysMsPerUe[ueId], percentile);
        }
        return values.str();
    }

    std::string GetPerUeQosSatisfactionRate(double targetDelayMs) const
    {
        std::ostringstream values;
        values << std::fixed << std::setprecision(4);
        for (uint32_t ueId = 0; ueId < m_delaysMsPerUe.size(); ++ueId)
        {
            if (ueId > 0)
            {
                values << ";";
            }
            values << CalculateQosSatisfactionRate(m_delaysMsPerUe[ueId], targetDelayMs);
        }
        return values.str();
    }

private:
    static double CalculateQosSatisfactionRate(const std::vector<double>& delaysMs,
                                               double targetDelayMs)
    {
        if (delaysMs.empty()) return 0.0;
        uint64_t count = 0;
        for (double d : delaysMs)
        {
            if (d <= targetDelayMs) count++;
        }
        return (static_cast<double>(count) / delaysMs.size()) * 100.0;
    }

    static double CalculatePercentile(const std::vector<double>& delaysMs, double percentile)
    {
        if (delaysMs.empty()) return 0.0;
        NS_ABORT_MSG_IF(percentile < 0.0 || percentile > 1.0,
                        "Delay percentile must be inside [0, 1]");
        std::vector<double> sortedDelays = delaysMs;
        std::sort(sortedDelays.begin(), sortedDelays.end());
        size_t index = static_cast<size_t>(percentile * sortedDelays.size());
        if (index >= sortedDelays.size()) index = sortedDelays.size() - 1;
        return sortedDelays[index];
    }

    std::unordered_map<uint64_t, std::pair<Time, uint32_t>> m_txInfo;
    std::vector<double> m_delaysMs;
    std::vector<std::vector<double>> m_delaysMsPerUe;
    std::vector<std::vector<double>> m_windowDelaysMsPerUe;
    std::vector<double> m_windowDelaysMs;
    uint64_t m_pendingBytes{0};
    uint64_t m_windowTxBytes{0};
    uint64_t m_windowRxBytes{0};
    uint64_t m_totalTxBytes;
    uint64_t m_totalRxBytes;
    uint64_t m_totalDropBytes;
    std::map<std::pair<uint16_t, uint8_t>, std::pair<uint64_t, uint64_t>> m_dropByBearer;
};

void
RecordRxForUeTrace(PacketDelayTracker* tracker,
                   uint32_t ueId,
                   Ptr<const Packet> packet,
                   const Address& from,
                   const Address& to)
{
    tracker->RecordRxForUe(ueId, packet, from, to);
}

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

    void SetGnbEnergyModel(Ptr<NrGnbEnergyModel> energyModel)
    {
        NS_ABORT_MSG_IF(!energyModel, "Cell-DTX requires an NrGnbEnergyModel");
        m_gnbEnergyModel = energyModel;
    }

    void UpdateState(double activeWindowMs, BwpState bwp)
    {
        m_activeWindowMs = std::max(0.0, std::min(m_periodMs, activeWindowMs));
        m_bwp = bwp;
        ConfigureSchedulerEligibility();
    }

    void ConfigurePhysicalBwpResources(
        const std::vector<Ptr<NrMacSchedulerNs3>>& schedulers,
        const std::vector<Ptr<NrGnbPhy>>& phys)
    {
        NS_ABORT_MSG_IF(schedulers.size() != phys.size() || schedulers.size() < 2,
                        "Physical BWP control requires matching scheduler/PHY vectors");
        for (uint32_t i = 0; i < schedulers.size(); ++i)
        {
            NS_ABORT_MSG_IF(!schedulers[i] || !phys[i],
                            "Missing scheduler or PHY for physical BWP " << i);
        }
        m_physicalBwpSchedulers = schedulers;
        m_physicalBwpPhys = phys;
        ConfigureSchedulerEligibility();
    }

    uint64_t GetDlRlcBacklogBytes() const
    {
        return GetActiveScheduler()->GetTotalDlBufferBytes();
    }

    NrMacSchedulerNs3::DlObservationStats GetDlObservationStats() const
    {
        return GetActiveScheduler()->GetDlObservationStats();
    }

    std::vector<uint32_t> GetPerUeDlBufferBytes() const
    {
        std::vector<uint32_t> values;
        const auto add = [&values](const Ptr<NrMacSchedulerNs3>& scheduler) {
            const auto stats = scheduler->GetDlObservationStats();
            if (values.size() < stats.perUeBufferBytes.size())
            {
                values.resize(stats.perUeBufferBytes.size(), 0);
            }
            for (uint32_t i = 0; i < stats.perUeBufferBytes.size(); ++i)
            {
                values[i] += stats.perUeBufferBytes[i];
            }
        };
        if (m_physicalBwpSchedulers.empty())
        {
            add(GetActiveScheduler());
        }
        else
        {
            for (const auto& scheduler : m_physicalBwpSchedulers) add(scheduler);
        }
        return values;
    }

    std::pair<double, double> GetDlQueueOccupancyRatios() const
    {
        constexpr double kRlcBearerCapacityBytes = 10.0 * 1024.0;
        double totalUsedBytes = 0.0;
        double maxBearerRatio = 0.0;
        for (const auto& stats : GetActiveScheduler()->GetDlBearerBufferStats())
        {
            const double usedBytes = static_cast<double>(stats.txQueueBytes) +
                                     static_cast<double>(stats.retxQueueBytes) +
                                     static_cast<double>(stats.statusPduBytes);
            totalUsedBytes += usedBytes;
            maxBearerRatio = std::max(maxBearerRatio, usedBytes / kRlcBearerCapacityBytes);
        }
        const double totalCapacityBytes =
            static_cast<double>(GetActiveScheduler()->GetDlBearerCount()) *
            kRlcBearerCapacityBytes;
        const double totalRatio = totalCapacityBytes > 0.0
                                      ? totalUsedBytes / totalCapacityBytes
                                      : 0.0;
        return {totalRatio, maxBearerRatio};
    }

    bool IsCellActiveAt(Time time) const
    {
        if (!m_enableGating || m_activeWindowMs >= m_periodMs)
        {
            return true;
        }
        const int64_t periodNs = MilliSeconds(m_periodMs).GetNanoSeconds();
        int64_t phaseNs = time.GetNanoSeconds() % periodNs;
        if (phaseNs < 0)
        {
            phaseNs += periodNs;
        }
        return phaseNs < MilliSeconds(m_activeWindowMs).GetNanoSeconds();
    }

private:
    Ptr<NrMacSchedulerNs3> GetActiveScheduler() const
    {
        if (!m_physicalBwpSchedulers.empty())
        {
            const uint32_t index = static_cast<uint32_t>(m_bwp);
            NS_ABORT_MSG_IF(index >= m_physicalBwpSchedulers.size(),
                            "Active physical BWP index is out of range");
            return m_physicalBwpSchedulers[index];
        }
        return m_scheduler;
    }

    void BeginCycle()
    {
        ApplyBwpMask();
        ConfigureSchedulerEligibility();
        if (m_gnbEnergyModel)
        {
            m_gnbEnergyModel->ChangeState(static_cast<int>(NrGnbPowerState::MicroSleep));
        }
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
        if (!m_gnbEnergyModel)
        {
            return;
        }
        const Time idleDuration = MilliSeconds(m_periodMs - m_activeWindowMs);
        const NrGnbPowerState sleepState = m_gnbEnergyModel->SelectSleepState(idleDuration);
        m_gnbEnergyModel->ChangeState(static_cast<int>(sleepState));
    }

    void ConfigureSchedulerEligibility()
    {
        const Time period = MilliSeconds(m_periodMs);
        const Time activeWindow = MilliSeconds(m_activeWindowMs);
        auto configure = [&](const Ptr<NrMacSchedulerNs3>& scheduler) {
            scheduler->SetDlDataSchedulingEnabled(true);
            if (m_enableGating)
            {
                scheduler->ConfigureCellDtx(period, activeWindow);
            }
            else
            {
                scheduler->DisableCellDtx();
            }
        };
        auto configurePhy = [&](const Ptr<NrGnbPhy>& phy) {
            if (m_enableGating)
            {
                phy->ConfigureCellDtx(period, activeWindow);
            }
            else
            {
                phy->DisableCellDtx();
            }
        };

        if (m_physicalBwpSchedulers.empty())
        {
            configure(m_scheduler);
            configurePhy(m_gnbPhy);
            return;
        }
        for (uint32_t i = 0; i < m_physicalBwpSchedulers.size(); ++i)
        {
            configure(m_physicalBwpSchedulers[i]);
            configurePhy(m_physicalBwpPhys[i]);
        }
    }

    void ApplyBwpMask()
    {
        if (!m_physicalBwpSchedulers.empty())
        {
            // Each real BWP PHY already exposes exactly its own bandwidth.
            // Do not apply the legacy notched-RBG approximation on top.
            return;
        }
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
    Ptr<NrGnbEnergyModel> m_gnbEnergyModel;
    std::vector<Ptr<NrMacSchedulerNs3>> m_physicalBwpSchedulers;
    std::vector<Ptr<NrGnbPhy>> m_physicalBwpPhys;
    double m_periodMs;
    bool m_enableGating;
    bool m_enableBwpDynamic;
    double m_activeWindowMs;
    BwpState m_bwp;
};

class EligibilityOverlapTracker
{
public:
    EligibilityOverlapTracker(const std::vector<Ptr<NrUeDrxModel>>& drxModels,
                              const RadioResourceController* radioController,
                              Time sampleInterval,
                              Time stopTime)
        : m_drxModels(drxModels),
          m_radioController(radioController),
          m_sampleInterval(sampleInterval),
          m_stopTime(stopTime),
          m_awakeTimeS(drxModels.size(), 0.0),
          m_awakeCellInactiveTimeS(drxModels.size(), 0.0)
    {
        NS_ABORT_MSG_IF(!m_radioController, "Eligibility tracker needs a radio controller");
        NS_ABORT_MSG_IF(m_sampleInterval <= Seconds(0),
                        "Eligibility tracker sample interval must be positive");
    }

    void Start(Time startTime)
    {
        NS_ABORT_MSG_IF(startTime >= m_stopTime,
                        "Eligibility tracker start must precede its stop time");
        Simulator::Schedule(startTime - Simulator::Now(),
                            &EligibilityOverlapTracker::Sample,
                            this);
    }

    double GetAwakePct() const
    {
        return GetAggregatePct(m_awakeTimeS, m_totalTimeS * m_drxModels.size());
    }

    double GetAwakeCellInactivePct() const
    {
        return GetAggregatePct(m_awakeCellInactiveTimeS,
                               m_totalTimeS * m_drxModels.size());
    }

    double GetWasteWithinAwakePct() const
    {
        double awakeTimeS = 0.0;
        double wasteTimeS = 0.0;
        for (uint32_t ueId = 0; ueId < m_drxModels.size(); ++ueId)
        {
            awakeTimeS += m_awakeTimeS[ueId];
            wasteTimeS += m_awakeCellInactiveTimeS[ueId];
        }
        return GetAggregatePct(wasteTimeS, awakeTimeS);
    }

    std::string GetAwakeCellInactivePctPerUe() const
    {
        std::ostringstream values;
        values << std::fixed << std::setprecision(3);
        for (uint32_t ueId = 0; ueId < m_drxModels.size(); ++ueId)
        {
            if (ueId > 0)
            {
                values << ";";
            }
            values << GetAggregatePct(m_awakeCellInactiveTimeS[ueId], m_totalTimeS);
        }
        return values.str();
    }

    std::string GetWasteWithinAwakePctPerUe() const
    {
        std::ostringstream values;
        values << std::fixed << std::setprecision(3);
        for (uint32_t ueId = 0; ueId < m_drxModels.size(); ++ueId)
        {
            if (ueId > 0)
            {
                values << ";";
            }
            values << GetAggregatePct(m_awakeCellInactiveTimeS[ueId], m_awakeTimeS[ueId]);
        }
        return values.str();
    }

private:
    static double GetAggregatePct(double numerator, double denominator)
    {
        return denominator > 0.0 ? 100.0 * numerator / denominator : 0.0;
    }

    static double GetAggregatePct(const std::vector<double>& values, double denominator)
    {
        double numerator = 0.0;
        for (double value : values)
        {
            numerator += value;
        }
        return GetAggregatePct(numerator, denominator);
    }

    void Sample()
    {
        const Time now = Simulator::Now();
        if (now >= m_stopTime)
        {
            return;
        }
        const Time interval = std::min(m_sampleInterval, m_stopTime - now);
        const double intervalS = interval.GetSeconds();
        const bool cellActive = m_radioController->IsCellActiveAt(now);
        for (uint32_t ueId = 0; ueId < m_drxModels.size(); ++ueId)
        {
            const bool ueAwake = !m_drxModels[ueId] || m_drxModels[ueId]->IsActiveAt(now);
            if (ueAwake)
            {
                m_awakeTimeS[ueId] += intervalS;
                if (!cellActive)
                {
                    m_awakeCellInactiveTimeS[ueId] += intervalS;
                }
            }
        }
        m_totalTimeS += intervalS;
        Simulator::Schedule(interval, &EligibilityOverlapTracker::Sample, this);
    }

    std::vector<Ptr<NrUeDrxModel>> m_drxModels;
    const RadioResourceController* m_radioController;
    Time m_sampleInterval;
    Time m_stopTime;
    std::vector<double> m_awakeTimeS;
    std::vector<double> m_awakeCellInactiveTimeS;
    double m_totalTimeS{0.0};
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
                         double queueWarnRatio = 0.50, double queueHighRatio = 0.80,
                         double bwpDwellMs = 100.0,
                         std::string timelineCsvName = "active_window_timeline.csv",
                         double warmupMs = 200.0,
                         BwpState initialBwp = BWP_NARROW_20MHZ,
                         bool enableTrueBwpControl = false,
                         bool enablePerUeAction = false,
                         NetDeviceContainer trueBwpUeDevices = NetDeviceContainer(),
                         energy::DeviceEnergyModelContainer trueBwpUeEnergyModels =
                             energy::DeviceEnergyModelContainer())
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
          m_currentBwp(initialBwp),
          m_bwpInactivityTimerMs(0.0),
          m_bwpInactivityTimeoutMs(10.0),
          m_cellDtxPeriodMs(40.0),
          m_avgTbsPerSlot(3000),
          m_slotDurationMs(0.5),
          m_queueWarnRatio(queueWarnRatio),
          m_queueHighRatio(queueHighRatio),
          m_bwpDwellMs(std::max(0.0, bwpDwellMs)),
          m_bwpDwellRemainingMs(0.0),
          m_timelineCsvName(timelineCsvName),
          m_warmupMs(warmupMs),
          m_enableTrueBwpControl(enableTrueBwpControl),
          m_enablePerUeAction(enablePerUeAction),
          m_trueBwpUeDevices(trueBwpUeDevices),
          m_trueBwpUeEnergyModels(trueBwpUeEnergyModels),
          m_ipcClient(nullptr)
    {
        m_perUeBwp.assign(m_trueBwpUeDevices.GetN(), BWP_NARROW_20MHZ);
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
            csv << "SimTimeSec,ActiveWindowMs,CurrentBwp,TotalPendingBytes,RlcBacklogBytes,RlcTotalOccupancyPct,RlcMaxBearerOccupancyPct,DataRateMbps\n";
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
    void ApplyPhysicalBwpSwitch(BwpState oldBwp)
    {
        if (m_enableTrueBwpControl && m_currentBwp != oldBwp)
        {
            SwitchAllUesToPrimaryBwpAndEnergy(
                m_trueBwpUeDevices,
                m_trueBwpUeEnergyModels,
                static_cast<uint8_t>(m_currentBwp));
        }
    }

    void ApplyPerUeBwpStates(const std::vector<BwpState>& nextStates)
    {
        NS_ABORT_MSG_IF(nextStates.size() != m_perUeBwp.size(),
                        "Per-UE BWP action size does not match UE count");
        for (uint32_t ueId = 0; ueId < nextStates.size(); ++ueId)
        {
            if (nextStates[ueId] != m_perUeBwp[ueId])
            {
                SwitchUeToPrimaryBwpAndEnergy(m_trueBwpUeDevices,
                                               m_trueBwpUeEnergyModels,
                                               ueId,
                                               static_cast<uint8_t>(nextStates[ueId]));
                m_perUeBwp[ueId] = nextStates[ueId];
            }
        }
    }

    void AppendTimeline(uint64_t totalPending, uint64_t rlcBacklogBytes,
                        double totalQueueRatio, double maxBearerRatio)
    {
        std::ofstream csv(m_timelineCsvName, std::ios::app);
        if (csv.is_open())
        {
            csv << Simulator::Now().GetSeconds() << ","
                << m_currentActiveMs << ","
                << (m_currentBwp == BWP_WIDE_100MHZ ? 100 : 20) << ","
                << totalPending << ","
                << rlcBacklogBytes << ","
                << (100.0 * totalQueueRatio) << ","
                << (100.0 * maxBearerRatio) << ","
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
    double m_queueWarnRatio;
    double m_queueHighRatio;
    double m_bwpDwellMs;
    double m_bwpDwellRemainingMs;
    std::string m_timelineCsvName;
    double m_warmupMs;
    bool m_enableTrueBwpControl;
    bool m_enablePerUeAction;
    NetDeviceContainer m_trueBwpUeDevices;
    energy::DeviceEnergyModelContainer m_trueBwpUeEnergyModels;
    std::vector<BwpState> m_perUeBwp;
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
    const auto queueRatios = m_radioController
                                 ? m_radioController->GetDlQueueOccupancyRatios()
                                 : std::pair<double, double>{0.0, 0.0};
    const double totalQueueRatio = queueRatios.first;
    const double maxBearerRatio = queueRatios.second;
    m_bwpDwellRemainingMs = std::max(0.0, m_bwpDwellRemainingMs - m_evalIntervalSec * 1000.0);
    const bool bwpSwitchAllowed = m_bwpDwellRemainingMs <= 1e-9;

    double oldActiveMs = m_currentActiveMs;
    BwpState oldBwp = m_currentBwp;

    if (maxBearerRatio >= m_queueHighRatio || totalQueueRatio >= m_queueHighRatio)
    {
        m_currentActiveMs = std::min(m_maxActiveMs, m_currentActiveMs + m_stepUpMs);

        if (m_enableBwpDynamic && bwpSwitchAllowed)
        {
            if (m_currentBwp != BWP_WIDE_100MHZ)
            {
                m_currentBwp = BWP_WIDE_100MHZ;
                m_bwpDwellRemainingMs = m_bwpDwellMs;
            }
            m_bwpInactivityTimerMs = m_bwpInactivityTimeoutMs;
        }
    }
    else if (maxBearerRatio < m_queueWarnRatio && totalQueueRatio < m_queueWarnRatio)
    {
        m_currentActiveMs = std::max(m_minActiveMs, m_currentActiveMs - m_stepDownMs);

        if (m_enableBwpDynamic)
        {
            if (m_bwpInactivityTimerMs <= 0.0 && bwpSwitchAllowed)
            {
                if (m_currentBwp != BWP_NARROW_20MHZ)
                {
                    m_currentBwp = BWP_NARROW_20MHZ;
                    m_bwpDwellRemainingMs = m_bwpDwellMs;
                }
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
        ApplyPhysicalBwpSwitch(oldBwp);

        if (m_enableDciDynamic)
        {
            if (m_energyMetrics) m_energyMetrics->UpdateDynamicOnDurationMs(m_currentActiveMs);
        }
    }

    AppendTimeline(totalPending, rlcBacklogBytes, totalQueueRatio, maxBearerRatio);
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
    const auto queueRatios = m_radioController
                                 ? m_radioController->GetDlQueueOccupancyRatios()
                                 : std::pair<double, double>{0.0, 0.0};
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
    const auto perUeBufferBytes = m_radioController
                                      ? m_radioController->GetPerUeDlBufferBytes()
                                      : std::vector<uint32_t>{};
    const auto perUeDelayMs = m_tracker
                                  ? m_tracker->GetWindowP99LatencyPerUe()
                                  : std::vector<double>{};

    if (m_tracker) m_tracker->ResetWindowStats();
    if (m_energyMetrics) m_energyMetrics->ResetWindowStats();
    constexpr double absoluteActionStepMs = 2.0;

    if (m_enablePerUeAction)
    {
        int activeActionIndex = -1;
        std::vector<BwpState> nextPerUeBwp;
        if (!m_ipcClient->StepIpcS5((double)rlcBacklogBytes,
                                    (double)dlStats.maxUeBufferBytes,
                                    queueRatios.first,
                                    queueRatios.second,
                                    offeredRateMbps,
                                    deliveredRateMbps,
                                    p99DelayMs,
                                    dlStats.meanCqi,
                                    dlStats.p10Cqi,
                                    dlStats.minCqi,
                                    dlStats.meanMcs,
                                    m_currentActiveMs,
                                    m_perUeBwp,
                                    perUeBufferBytes,
                                    perUeDelayMs,
                                    gnbPowerW,
                                    uePowerMw,
                                    jainFairness,
                                    activeActionIndex,
                                    nextPerUeBwp))
        {
            Simulator::Stop();
            return;
        }
        const int activeActionCount =
            static_cast<int>(std::llround((m_maxActiveMs - m_minActiveMs) /
                                          absoluteActionStepMs)) +
            1;
        if (activeActionIndex < 0 || activeActionIndex >= activeActionCount)
        {
            Simulator::Stop();
            return;
        }
        m_currentActiveMs = std::max(
            m_minActiveMs,
            std::min(m_maxActiveMs,
                     m_minActiveMs + activeActionIndex * absoluteActionStepMs));
        ApplyPerUeBwpStates(nextPerUeBwp);
        if (m_radioController)
        {
            m_radioController->UpdateState(m_currentActiveMs, BWP_NARROW_20MHZ);
        }
        AppendTimeline(totalPending, rlcBacklogBytes, queueRatios.first, queueRatios.second);
        Simulator::Schedule(Seconds(m_evalIntervalSec),
                             &DynamicDtxController::EvaluateAndAdjustRlIpc,
                             this);
        return;
    }

    int actionIdx = m_ipcClient->StepIpc((double)rlcBacklogBytes,
                                         (double)dlStats.maxUeBufferBytes,
                                         queueRatios.first,
                                         queueRatios.second,
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

    const int activeActionCount =
        static_cast<int>(std::llround((m_maxActiveMs - m_minActiveMs) /
                                      absoluteActionStepMs)) +
        1;
    const int actionCount = activeActionCount * 2;
    if (actionIdx < 0 || actionIdx >= actionCount)
    {
        Simulator::Stop();
        return;
    }

    if (actionIdx >= 0)
    {
        BwpState oldBwp = m_currentBwp;
        const int activeActionIndex = actionIdx / 2;
        const double targetActiveMs =
            m_minActiveMs + activeActionIndex * absoluteActionStepMs;
        BwpState nextBwp = m_enableBwpDynamic
                               ? ((actionIdx % 2 == 1) ? BWP_WIDE_100MHZ
                                                     : BWP_NARROW_20MHZ)
                               : m_currentBwp;

        m_currentActiveMs = std::max(
            m_minActiveMs, std::min(m_maxActiveMs, targetActiveMs));
        m_currentBwp = nextBwp;

        if (m_radioController)
        {
            m_radioController->UpdateState(m_currentActiveMs, m_currentBwp);
        }
        ApplyPhysicalBwpSwitch(oldBwp);

        if (m_enableDciDynamic)
        {
            if (m_energyMetrics) m_energyMetrics->UpdateDynamicOnDurationMs(m_currentActiveMs);
        }
    }

    AppendTimeline(totalPending, rlcBacklogBytes, queueRatios.first, queueRatios.second);

    Simulator::Schedule(Seconds(m_evalIntervalSec), &DynamicDtxController::EvaluateAndAdjustRlIpc, this);
}

void ExportStatsToCsv(const std::string &filename, double simTime, uint32_t numUes,
                       bool enableGating, bool enableQueueAware,
                       double cellDtxPeriodMs, double cellDtxActiveMs, double udpDataRate,
                       uint64_t totalRxBytes, uint64_t trackerDropBytes, uint64_t pendingBytes,
                       uint64_t pendingAtTrafficStop,
                       uint64_t rlcTxDropPackets, uint64_t rlcTxDropBytes,
                       uint64_t harqNackEvents, uint32_t harqMaxRetx,
                       uint64_t unattributedFlowLostPackets,
                       uint64_t flowTxPackets, uint64_t flowRxPackets, uint64_t flowLostPackets,
                       uint64_t flowTxBytes, uint64_t flowRxBytes,
                       double avgPowerWatts, double totalEnergyJoules,
                       double energyEfficiency, double p99LatencyMs, double qosSatisfactionRate,
                       bool enableCdrx = false, bool enableDciDynamic = false, bool enableBwpDynamic = false,
                       double phaseOffsetMs = 0.0, const std::string& ueCdrxOffsetsMs = "",
                       uint32_t cdrxInactivityMs = 0,
                       const std::string& ueCdrxInactivityMs = "",
                       double ueAvgPowerMw = 0.0, double ueTotalEnergyJoules = 0.0,
                       double ueBatteryMah = 0.0, double ue3EnergyJoules = 0.0,
                       double p50LatencyMs = 0.0, double p95LatencyMs = 0.0,
                       double qos20Pct = 0.0, double qos30Pct = 0.0, double qos50Pct = 0.0,
                       double ueAwakePct = 0.0, double ueAwakeCellInactivePct = 0.0,
                       double wasteWithinUeAwakePct = 0.0,
                       const std::string& ueAwakeCellInactivePctPerUe = "",
                       const std::string& wasteWithinUeAwakePctPerUe = "",
                       const std::string& perUeP99LatencyMs = "",
                       const std::string& perUeQos20Pct = "",
                       const std::string& perUeQos30Pct = "",
                       const std::string& perUeQos50Pct = "")
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
        csvFile << "SimTime,NumUes,EnableGating,EnableQueueAware,EnableCdrx,EnableDciDynamic,EnableBwpDynamic,PhaseOffsetMs,UeCdrxOffsetsMs,CdrxInactivityMs,UeCdrxInactivityMs,CellDtxPeriodMs,CellDtxActiveMs,UdpDataRateMbps,TotalOfferedRateMbps,TotalRxBytes,TrackerDropBytes,PendingBytes,PendingBytesAtTrafficStop,RlcTxDropPackets,RlcTxDropBytes,HarqNackEvents,HarqMaxRetx,UnattributedFlowLostPackets,FlowTxPackets,FlowRxPackets,FlowLostPackets,FlowTxBytes,FlowRxBytes,TotalRxMbits,AvgPowerWatts,TotalEnergyJoules,EnergyEfficiency_MbitsPerJoule,P99LatencyMs,QosSatisfactionRatePct,UeAvgPowerMw,UeTotalEnergyJoules,UeBatteryMah,Ue3EnergyJoules,P50LatencyMs,P95LatencyMs,Qos20Pct,Qos30Pct,Qos50Pct,UeAwakePct,UeAwakeCellInactivePct,WasteWithinUeAwakePct,UeAwakeCellInactivePctPerUe,WasteWithinUeAwakePctPerUe,PerUeP99LatencyMs,PerUeQos20Pct,PerUeQos30Pct,PerUeQos50Pct\n";
    }

    double totalRxMbits = (totalRxBytes * 8.0) / 1e6;

    csvFile << simTime << ","
            << numUes << ","
            << (enableGating ? 1 : 0) << ","
            << (enableQueueAware ? 1 : 0) << ","
            << (enableCdrx ? 1 : 0) << ","
            << (enableDciDynamic ? 1 : 0) << ","
            << (enableBwpDynamic ? 1 : 0) << ","
            << phaseOffsetMs << ","
            << "\"" << ueCdrxOffsetsMs << "\"" << ","
            << cdrxInactivityMs << ","
            << "\"" << ueCdrxInactivityMs << "\"" << ","
            << cellDtxPeriodMs << ","
            << cellDtxActiveMs << ","
            << udpDataRate << ","
            << numUes * udpDataRate << ","
            << totalRxBytes << ","
            << trackerDropBytes << ","
            << pendingBytes << ","
            << pendingAtTrafficStop << ","
            << rlcTxDropPackets << ","
            << rlcTxDropBytes << ","
            << harqNackEvents << ","
            << harqMaxRetx << ","
            << unattributedFlowLostPackets << ","
            << flowTxPackets << ","
            << flowRxPackets << ","
            << flowLostPackets << ","
            << flowTxBytes << ","
            << flowRxBytes << ","
            << totalRxMbits << ","
            << avgPowerWatts << ","
            << totalEnergyJoules << ","
            << energyEfficiency << ","
            << p99LatencyMs << ","
            << qosSatisfactionRate << ","
            << ueAvgPowerMw << ","
            << ueTotalEnergyJoules << ","
            << ueBatteryMah << ","
            << ue3EnergyJoules << ","
            << p50LatencyMs << ","
            << p95LatencyMs << ","
            << qos20Pct << ","
            << qos30Pct << ","
            << qos50Pct << ","
            << ueAwakePct << ","
            << ueAwakeCellInactivePct << ","
            << wasteWithinUeAwakePct << ","
            << "\"" << ueAwakeCellInactivePctPerUe << "\"" << ","
            << "\"" << wasteWithinUeAwakePctPerUe << "\"" << ","
            << "\"" << perUeP99LatencyMs << "\"" << ","
            << "\"" << perUeQos20Pct << "\"" << ","
            << "\"" << perUeQos30Pct << "\"" << ","
            << "\"" << perUeQos50Pct << "\"" << "\n";

    csvFile.close();
}

void
PrintDlBearerBufferStats(const Ptr<NrMacSchedulerNs3>& scheduler, const char* phase)
{
    constexpr uint32_t kRlcUmCapacityBytes = 10 * 1024;
    for (const auto& stats : scheduler->GetDlBearerBufferStats())
    {
        const uint32_t usedBytes = stats.txQueueBytes + stats.retxQueueBytes + stats.statusPduBytes;
        const uint32_t remainingBytes = usedBytes < kRlcUmCapacityBytes
                                             ? kRlcUmCapacityBytes - usedBytes
                                             : 0;
        std::cout << "RlcBearerBuffer phase=" << phase
                  << ",Rnti=" << stats.rnti
                  << ",Lcid=" << static_cast<uint32_t>(stats.lcid)
                  << ",TxBytes=" << stats.txQueueBytes
                  << ",RetxBytes=" << stats.retxQueueBytes
                  << ",StatusBytes=" << stats.statusPduBytes
                  << ",UsedBytes=" << usedBytes
                  << ",CapacityBytes=" << kRlcUmCapacityBytes
                  << ",RemainingBytes=" << remainingBytes << std::endl;
    }
}

class RlcBearerTelemetryProbe
{
  public:
    RlcBearerTelemetryProbe(Ptr<NrMacSchedulerNs3> scheduler,
                            std::string filename,
                            double sampleIntervalMs,
                            double stopTimeSec)
        : m_scheduler(std::move(scheduler)),
          m_filename(std::move(filename)),
          m_interval(Seconds(sampleIntervalMs / 1000.0)),
          m_stopTime(Seconds(stopTimeSec))
    {
        NS_ABORT_MSG_IF(!m_scheduler, "RLC bearer telemetry requires a scheduler");
        NS_ABORT_MSG_IF(sampleIntervalMs <= 0.0,
                        "RLC bearer telemetry sample interval must be positive");
    }

    void Start()
    {
        m_csv.open(m_filename, std::ios::trunc);
        NS_ABORT_MSG_IF(!m_csv.is_open(),
                        "Unable to open RLC bearer telemetry " << m_filename);
        m_csv << "SimTimeSec,Rnti,Lcid,TxBytes,RetxBytes,StatusBytes,UsedBytes,"
                 "CapacityBytes,RemainingBytes,PeakUsedBytes\n";
        Simulator::ScheduleNow(&RlcBearerTelemetryProbe::Sample, this);
    }

    void PrintPeaks() const
    {
        for (const auto& [key, peak] : m_peakUsedBytes)
        {
            std::cout << "RlcBearerPeak,Rnti=" << key.first
                      << ",Lcid=" << static_cast<uint32_t>(key.second)
                      << ",PeakUsedBytes=" << peak
                      << ",CapacityBytes=" << kCapacityBytes
                      << ",PeakRemainingBytes=" << (peak < kCapacityBytes
                                                             ? kCapacityBytes - peak
                                                             : 0)
                      << std::endl;
        }
    }

  private:
    static constexpr uint32_t kCapacityBytes = 10 * 1024;

    void Sample()
    {
        for (const auto& stats : m_scheduler->GetDlBearerBufferStats())
        {
            const uint32_t usedBytes = stats.txQueueBytes + stats.retxQueueBytes +
                                       stats.statusPduBytes;
            auto& peak = m_peakUsedBytes[{stats.rnti, stats.lcid}];
            peak = std::max(peak, usedBytes);
            const uint32_t remainingBytes = usedBytes < kCapacityBytes
                                                 ? kCapacityBytes - usedBytes
                                                 : 0;
            m_csv << std::fixed << std::setprecision(6) << Simulator::Now().GetSeconds() << ","
                  << stats.rnti << "," << static_cast<uint32_t>(stats.lcid) << ","
                  << stats.txQueueBytes << "," << stats.retxQueueBytes << ","
                  << stats.statusPduBytes << "," << usedBytes << "," << kCapacityBytes << ","
                  << remainingBytes << "," << peak << "\n";
        }
        if (Simulator::Now() + m_interval < m_stopTime)
        {
            Simulator::Schedule(m_interval, &RlcBearerTelemetryProbe::Sample, this);
        }
    }

    Ptr<NrMacSchedulerNs3> m_scheduler;
    std::string m_filename;
    Time m_interval;
    Time m_stopTime;
    std::ofstream m_csv;
    std::map<std::pair<uint16_t, uint8_t>, uint32_t> m_peakUsedBytes;
};

class HarqDiagnostics
{
  public:
    void OnDlHarqFeedback(const DlHarqInfo& info)
    {
        auto& counters = m_byRnti[info.m_rnti];
        if (info.m_harqStatus == DlHarqInfo::ACK)
        {
            counters.acks++;
        }
        else
        {
            counters.nacks++;
        }
        counters.maxRetx = std::max<uint32_t>(counters.maxRetx, info.m_numRetx);
    }

    void Print() const
    {
        uint64_t totalAcks = 0;
        uint64_t totalNacks = 0;
        for (const auto& [rnti, counters] : m_byRnti)
        {
            totalAcks += counters.acks;
            totalNacks += counters.nacks;
            std::cout << "HarqDl,Rnti=" << rnti << ",Acks=" << counters.acks
                      << ",Nacks=" << counters.nacks
                      << ",MaxRetx=" << counters.maxRetx << std::endl;
        }
        std::cout << "HarqDlTotal,Acks=" << totalAcks << ",Nacks=" << totalNacks
                  << std::endl;
    }

    uint64_t GetTotalNacks() const
    {
        uint64_t total = 0;
        for (const auto& [rnti, counters] : m_byRnti)
        {
            total += counters.nacks;
        }
        return total;
    }

    uint32_t GetMaxRetx() const
    {
        uint32_t maximum = 0;
        for (const auto& [rnti, counters] : m_byRnti)
        {
            maximum = std::max(maximum, counters.maxRetx);
        }
        return maximum;
    }

  private:
    struct Counters
    {
        uint64_t acks{0};
        uint64_t nacks{0};
        uint32_t maxRetx{0};
    };
    std::map<uint16_t, Counters> m_byRnti;
};

int main(int argc, char *argv[])
{
    uint32_t numUes = 10;
    uint32_t numerology = 1;
    double simTime = 10.0;
    double trafficStopSec = -1.0;
    double rlcBearerSampleMs = 1.0;
    double udpPacketRateMbps = 2.0;
    bool enableBurstTraffic = false;
    bool synchronizeBurstTraffic = false;
    double udpBurstRateMbps = 3.0;
    double burstMeanOnMs = 80.0;
    double burstMeanOffMs = 120.0;
    double gnbTxPowerDbm = 30.0;
    std::string gnbEnergyBsCategory = "BsCat1";
    std::string gnbEnergyRefConfig = "Custom";
    double gnbEnergyReferenceTxPowerDbm = 30.0;
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
    bool enableTrueBwpSwitchScript = false;
    bool enablePerUeBwpSwitchScript = false;
    bool enableBwpValidationTraces = false;
    uint32_t fixedBwpMHz = 100;
    std::string perUeBwpMHz;
    uint32_t perUeBwpSwitchUeId = 1;
    double perUeBwpSwitchToWideSec = 0.8;
    double perUeBwpSwitchBackToNarrowSec = 1.4;
    double bwpSwitchToWideSec = 0.8;
    double bwpSwitchBackToNarrowSec = 1.4;
    bool enableRlIpc = false;
    bool enablePerUeBwpAction = false;
    int ipcPort = 5555;
    double phaseOffsetMs = 0.0;
    std::string ueCdrxOffsetsMs;
    std::string ueCdrxInactivityMs;
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
    double controllerQueueWarnRatio = 0.50;
    double controllerQueueHighRatio = 0.80;
    double controllerBwpDwellMs = 100.0;
    uint32_t rngSeed = 42;
    uint64_t rngRun = 1;
    std::string csvFileName = "joint_energy_results.csv";
    std::string timelineCsvName = "active_window_timeline.csv";
    std::string bwpHandoverTraceCsv;
    double bwpHandoverSampleMs = 0.25;
    std::string tbContinuityTraceCsv;

    CommandLine cmd(__FILE__);
    cmd.AddValue("numUes", "Number of UEs", numUes);
    cmd.AddValue("numerology", "NR numerology (1 gives 30 kHz SCS)", numerology);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("trafficStopSec",
                 "Optional absolute time to stop UDP clients before simulation end",
                 trafficStopSec);
    cmd.AddValue("rlcBearerSampleMs",
                 "Per-bearer RLC scheduler telemetry sampling interval in milliseconds",
                 rlcBearerSampleMs);
    cmd.AddValue("udpDataRate", "Constant/base UDP downlink rate in Mbps per UE", udpPacketRateMbps);
    cmd.AddValue("enableBurstTraffic", "Add an independent On/Off burst source for every UE", enableBurstTraffic);
    cmd.AddValue("synchronizeBurstTraffic", "Use a shared deterministic On/Off burst regime across UEs", synchronizeBurstTraffic);
    cmd.AddValue("udpBurstRate", "Additional UDP burst rate in Mbps per UE", udpBurstRateMbps);
    cmd.AddValue("burstMeanOnMs", "Mean burst On duration in ms", burstMeanOnMs);
    cmd.AddValue("burstMeanOffMs", "Mean burst Off duration in ms", burstMeanOffMs);
    cmd.AddValue("gnbTxPowerDbm", "gNB downlink transmission power in dBm", gnbTxPowerDbm);
    cmd.AddValue("gnbEnergyBsCategory", "TR 38.864 BS category (BsCat1 or BsCat2)", gnbEnergyBsCategory);
    cmd.AddValue("gnbEnergyRefConfig",
                 "TR 38.864 reference set (Custom uses the Set1 power row)",
                 gnbEnergyRefConfig);
    cmd.AddValue("gnbEnergyReferenceTxPowerDbm",
                 "Reference Tx power used to normalize sp when the energy reference set is Custom",
                 gnbEnergyReferenceTxPowerDbm);
    cmd.AddValue("gnbEnergyEtaMode", "TR 38.864 PA efficiency mode (Single or Dual)", gnbEnergyEtaMode);
    cmd.AddValue("gnbEnergyAntennaRatio", "TR 38.864 antenna dynamic-power fraction A", gnbEnergyAntennaRatio);
    cmd.AddValue("gnbEnergyPowerUnitW", "Absolute watts per TR 38.864 relative power unit", gnbEnergyPowerUnitW);
    cmd.AddValue("ueEnergyPowerUnitMw", "Absolute mW per TR 38.840 relative power unit", ueEnergyPowerUnitMw);
    cmd.AddValue("enableGating", "Enable Cell-DTX Scheduler Gating", enableGating);
    cmd.AddValue("enableQueueAware", "Enable Closed-Loop Dynamic Controller", enableQueueAware);
    cmd.AddValue("enableCdrx", "Enable UE C-DRX Energy Saving Mode", enableCdrx);
    cmd.AddValue("enableDciDynamic", "Enable 3GPP DCI-Level Dynamic Alignment", enableDciDynamic);
    cmd.AddValue("enableBwpDynamic",
                 "Enable policy-driven BWP selection: physical BWPs when "
                 "enableTrueBwp=true, otherwise legacy PRB profiles",
                 enableBwpDynamic);
    cmd.AddValue("enableTrueBwp",
                 "Experimental v6: create physical 20 and 100 MHz gNB/UE BWPs",
                 enableTrueBwp);
    cmd.AddValue("enableTrueBwpSwitchScript",
                 "Run deterministic physical-BWP sequence 20 -> 100 -> 20 MHz",
                 enableTrueBwpSwitchScript);
    cmd.AddValue("enablePerUeBwpSwitchScript",
                 "Run a deterministic per-UE physical-BWP sequence for one UE",
                 enablePerUeBwpSwitchScript);
    cmd.AddValue("enableBwpValidationTraces",
                 "Write the built-in UE PHY DCI trace for BWP validation",
                 enableBwpValidationTraces);
    cmd.AddValue("fixedBwpMHz",
                 "Fixed schedulable PRB profile (20 or 100 MHz-equivalent); the PHY remains 100 MHz",
                 fixedBwpMHz);
    cmd.AddValue("perUeBwpMHz",
                 "Optional comma-separated static per-UE physical BWP bandwidths (20 or 100 MHz)",
                 perUeBwpMHz);
    cmd.AddValue("perUeBwpSwitchUeId",
                 "UE index used by the deterministic per-UE BWP switch script",
                 perUeBwpSwitchUeId);
    cmd.AddValue("perUeBwpSwitchToWideSec",
                 "Simulation time for the selected UE's 20 -> 100 MHz switch",
                 perUeBwpSwitchToWideSec);
    cmd.AddValue("perUeBwpSwitchBackToNarrowSec",
                 "Simulation time for the selected UE's 100 -> 20 MHz switch",
                 perUeBwpSwitchBackToNarrowSec);
    cmd.AddValue("bwpSwitchToWideSec",
                 "Absolute simulation time for scripted 20 -> 100 MHz switch",
                 bwpSwitchToWideSec);
    cmd.AddValue("bwpSwitchBackToNarrowSec",
                 "Absolute simulation time for scripted 100 -> 20 MHz switch",
                 bwpSwitchBackToNarrowSec);
    cmd.AddValue("enableRlIpc", "Enable Socket IPC for Reinforcement Learning Co-Simulation", enableRlIpc);
    cmd.AddValue("enablePerUeBwpAction",
                 "Use S5 IPC with factorized per-UE BWP actions",
                 enablePerUeBwpAction);
    cmd.AddValue("ipcPort", "Socket IPC Server Port", ipcPort);
    cmd.AddValue("phaseOffsetMs", "Phase Offset between gNB and UE C-DRX in ms", phaseOffsetMs);
    cmd.AddValue("ueCdrxOffsetsMs",
                 "Comma-separated per-UE C-DRX offsets in ms; overrides phaseOffsetMs",
                 ueCdrxOffsetsMs);
    cmd.AddValue("cdrxOnDurationMs", "C-DRX OnDuration Timer in ms", cdrxOnDurationMs);
    cmd.AddValue("cdrxInactivityMs", "C-DRX Inactivity Timer in ms", cdrxInactivityMs);
    cmd.AddValue("ueCdrxInactivityMs",
                 "Comma-separated per-UE inactivity timers in ms; overrides cdrxInactivityMs",
                 ueCdrxInactivityMs);
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
    cmd.AddValue("controllerBacklogThresholdBytes", "Legacy fixed-byte rule threshold (accepted for compatibility; ignored)", controllerBacklogThresholdBytes);
    cmd.AddValue("controllerQueueWarnRatio", "Capacity-based rule warning occupancy ratio", controllerQueueWarnRatio);
    cmd.AddValue("controllerQueueHighRatio", "Capacity-based rule high occupancy ratio", controllerQueueHighRatio);
    cmd.AddValue("controllerBwpDwellMs", "Minimum time between rule-based BWP switches", controllerBwpDwellMs);
    cmd.AddValue("rngSeed", "ns-3 random-number generator seed", rngSeed);
    cmd.AddValue("rngRun", "ns-3 independent random-number generator run", rngRun);
    cmd.AddValue("csvFileName", "Output CSV file name", csvFileName);
    cmd.AddValue("timelineCsvName", "Controller timeline CSV file name", timelineCsvName);
    cmd.AddValue("bwpHandoverTraceCsv",
                 "Optional high-resolution scheduler backlog and BWP DL-symbol trace CSV",
                 bwpHandoverTraceCsv);
    cmd.AddValue("bwpHandoverSampleMs",
                 "Sampling interval in ms for bwpHandoverTraceCsv",
                 bwpHandoverSampleMs);
    cmd.AddValue("tbContinuityTraceCsv",
                 "Optional per-BWP DL TB TX/RX/HARQ continuity trace CSV",
                 tbContinuityTraceCsv);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(fixedBwpMHz != 20 && fixedBwpMHz != 100,
                    "fixedBwpMHz must be either 20 or 100");
    NS_ABORT_MSG_IF(enableTrueBwpSwitchScript && !enableTrueBwp,
                    "enableTrueBwpSwitchScript requires enableTrueBwp=true");
    NS_ABORT_MSG_IF(enablePerUeBwpSwitchScript &&
                        (!enableTrueBwp || enableBwpDynamic || enableTrueBwpSwitchScript),
                    "per-UE BWP switching requires true physical BWPs, no dynamic cell policy, and no all-UE script");
    NS_ABORT_MSG_IF(enablePerUeBwpSwitchScript && perUeBwpSwitchUeId >= numUes,
                    "perUeBwpSwitchUeId must be smaller than numUes");
    NS_ABORT_MSG_IF(enablePerUeBwpSwitchScript &&
                        (perUeBwpSwitchToWideSec <= std::max(0.02, controllerWarmupMs / 1000.0) ||
                         perUeBwpSwitchBackToNarrowSec <= perUeBwpSwitchToWideSec ||
                         perUeBwpSwitchBackToNarrowSec >= simTime),
                    "Per-UE BWP switch times must satisfy traffic start < wide < narrow < simTime");
    NS_ABORT_MSG_IF(enablePerUeBwpAction &&
                        (!enableTrueBwp || !enableRlIpc || enableBwpDynamic),
                    "enablePerUeBwpAction requires true physical BWPs, RL IPC, and no cell-level BWP policy");
    NS_ABORT_MSG_IF(!bwpHandoverTraceCsv.empty() && !enableTrueBwp,
                    "bwpHandoverTraceCsv requires enableTrueBwp=true");
    NS_ABORT_MSG_IF(!tbContinuityTraceCsv.empty() && !enableTrueBwpSwitchScript,
                    "tbContinuityTraceCsv currently requires enableTrueBwpSwitchScript=true");
    NS_ABORT_MSG_IF(enableTrueBwpSwitchScript && enableBwpDynamic,
                    "Choose either scripted or policy-driven physical BWP switching");
    NS_ABORT_MSG_IF(enableTrueBwpSwitchScript &&
                        (bwpSwitchToWideSec <= 0.0 ||
                         bwpSwitchBackToNarrowSec <= bwpSwitchToWideSec ||
                         bwpSwitchBackToNarrowSec >= simTime),
                    "Scripted BWP times must satisfy 0 < wide < narrow < simTime");
    NS_ABORT_MSG_IF(udpPacketRateMbps <= 0.0, "udpDataRate must be greater than zero");
    NS_ABORT_MSG_IF(trafficStopSec > 0.0 &&
                        (trafficStopSec <= std::max(0.02, controllerWarmupMs / 1000.0) ||
                         trafficStopSec >= simTime),
                    "trafficStopSec must be after traffic start and before simTime");
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
    NS_ABORT_MSG_IF(phaseOffsetMs < 0.0 || phaseOffsetMs >= cellDtxPeriodMs,
                    "phaseOffsetMs must satisfy 0 <= offset < C-DRX long cycle");

    std::vector<Time> ueCdrxStartOffsets;
    if (!ueCdrxOffsetsMs.empty())
    {
        NS_ABORT_MSG_IF(ueCdrxOffsetsMs.front() == ',' || ueCdrxOffsetsMs.back() == ',',
                        "ueCdrxOffsetsMs must not start or end with a comma");
        std::stringstream offsetList(ueCdrxOffsetsMs);
        std::string token;
        while (std::getline(offsetList, token, ','))
        {
            std::stringstream valueStream(token);
            double offsetMs;
            std::string extra;
            NS_ABORT_MSG_IF(!(valueStream >> offsetMs) || (valueStream >> extra),
                            "Invalid value in ueCdrxOffsetsMs: " << token);
            NS_ABORT_MSG_IF(offsetMs < 0.0 || offsetMs >= cellDtxPeriodMs,
                            "Every UE C-DRX offset must satisfy 0 <= offset < cycle");
            ueCdrxStartOffsets.push_back(MilliSeconds(offsetMs));
        }
        NS_ABORT_MSG_IF(ueCdrxStartOffsets.size() != numUes,
                        "ueCdrxOffsetsMs must contain exactly one value per UE");
        NS_ABORT_MSG_IF(!enableCdrx, "ueCdrxOffsetsMs requires enableCdrx=true");
    }
    std::vector<Time> ueCdrxInactivityTimers;
    if (!ueCdrxInactivityMs.empty())
    {
        NS_ABORT_MSG_IF(ueCdrxInactivityMs.front() == ',' ||
                            ueCdrxInactivityMs.back() == ',',
                        "ueCdrxInactivityMs must not start or end with a comma");
        std::stringstream timerList(ueCdrxInactivityMs);
        std::string token;
        while (std::getline(timerList, token, ','))
        {
            std::stringstream valueStream(token);
            double timerMs;
            std::string extra;
            NS_ABORT_MSG_IF(!(valueStream >> timerMs) || (valueStream >> extra),
                            "Invalid value in ueCdrxInactivityMs: " << token);
            NS_ABORT_MSG_IF(timerMs < 0.0,
                            "Every UE C-DRX inactivity timer must be non-negative");
            ueCdrxInactivityTimers.push_back(MilliSeconds(timerMs));
        }
        NS_ABORT_MSG_IF(ueCdrxInactivityTimers.size() != numUes,
                        "ueCdrxInactivityMs must contain exactly one value per UE");
        NS_ABORT_MSG_IF(!enableCdrx, "ueCdrxInactivityMs requires enableCdrx=true");
    }

    std::vector<uint32_t> perUeBwpMHzValues;
    if (!perUeBwpMHz.empty())
    {
        NS_ABORT_MSG_IF(!enableTrueBwp || enableBwpDynamic,
                        "perUeBwpMHz requires enableTrueBwp=true and enableBwpDynamic=false");
        NS_ABORT_MSG_IF(perUeBwpMHz.front() == ',' || perUeBwpMHz.back() == ',',
                        "perUeBwpMHz must not start or end with a comma");
        std::stringstream bwpList(perUeBwpMHz);
        std::string token;
        while (std::getline(bwpList, token, ','))
        {
            std::stringstream valueStream(token);
            uint32_t bwpMHz;
            std::string extra;
            NS_ABORT_MSG_IF(!(valueStream >> bwpMHz) || (valueStream >> extra),
                            "Invalid value in perUeBwpMHz: " << token);
            NS_ABORT_MSG_IF(bwpMHz != 20 && bwpMHz != 100,
                            "Every per-UE BWP must be either 20 or 100 MHz");
            perUeBwpMHzValues.push_back(bwpMHz);
        }
        NS_ABORT_MSG_IF(perUeBwpMHzValues.size() != numUes,
                        "perUeBwpMHz must contain exactly one value per UE");
    }

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

    const uint8_t selectedBwpId = enableTrueBwp && perUeBwpMHzValues.empty() &&
                                          !enableTrueBwpSwitchScript &&
                                          !enableBwpDynamic &&
                                          fixedBwpMHz == 100
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
    HarqDiagnostics harqDiagnostics;
    bool harqTraceConnected = true;
    for (uint32_t bwpId = 0; bwpId < allBwps.size(); ++bwpId)
    {
        Ptr<NrGnbMac> gnbMac = NrHelper::GetGnbMac(gnbNetDev.Get(0), bwpId);
        harqTraceConnected =
            gnbMac &&
            gnbMac->TraceConnectWithoutContext(
                "DlHarqFeedback",
                MakeCallback(&HarqDiagnostics::OnDlHarqFeedback, &harqDiagnostics)) &&
            harqTraceConnected;
    }
    std::cout << "DlHarqFeedbackTraceConnected=" << (harqTraceConnected ? 1 : 0)
              << std::endl;
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
                                            enableBwpDynamic && !enableTrueBwp);
    std::vector<Ptr<NrMacSchedulerNs3>> physicalSchedulers;
    std::vector<Ptr<NrGnbPhy>> physicalPhys;
    if (enableTrueBwp)
    {
        for (uint32_t bwpId = 0; bwpId < allBwps.size(); ++bwpId)
        {
            physicalSchedulers.push_back(DynamicCast<NrMacSchedulerNs3>(
                NrHelper::GetScheduler(gnbNetDev.Get(0), bwpId)));
            physicalPhys.push_back(NrHelper::GetGnbPhy(gnbNetDev.Get(0), bwpId));
        }
        radioController.ConfigurePhysicalBwpResources(physicalSchedulers, physicalPhys);
    }

    std::unique_ptr<BwpHandoverProbe> bwpHandoverProbe;
    if (!bwpHandoverTraceCsv.empty())
    {
        bwpHandoverProbe = std::make_unique<BwpHandoverProbe>(physicalSchedulers,
                                                              physicalPhys,
                                                              bwpHandoverTraceCsv,
                                                              bwpHandoverSampleMs);
        bwpHandoverProbe->Start(selectedBwpId);
    }

    std::unique_ptr<TbContinuityProbe> tbContinuityProbe;
    if (!tbContinuityTraceCsv.empty())
    {
        tbContinuityProbe = std::make_unique<TbContinuityProbe>(gnbNetDev,
                                                                ueNetDev,
                                                                allBwps.size(),
                                                                tbContinuityTraceCsv);
        tbContinuityProbe->Start(selectedBwpId);
    }

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

    PacketDelayTracker delayTracker(numUes);
    uint64_t pendingAtTrafficStop = 0;
    Ptr<Ipv4L3Protocol> remoteHostIpv4 = remoteHost->GetObject<Ipv4L3Protocol>();
    if (remoteHostIpv4)
    {
        remoteHostIpv4->TraceConnectWithoutContext("Tx", MakeCallback(&PacketDelayTracker::RecordTx, &delayTracker));
    }

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
    if (enableBwpValidationTraces)
    {
        nrHelper->EnableUePhyCtrlMsgsTraces();
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
            udpServer->TraceConnectWithoutContext(
                "RxWithAddresses",
                MakeBoundCallback(&RecordRxForUeTrace, &delayTracker, i));
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
    std::unique_ptr<RlcBearerTelemetryProbe> rlcBearerProbe;
    if (!csvFileName.empty())
    {
        rlcBearerProbe = std::make_unique<RlcBearerTelemetryProbe>(
            scheduler, csvFileName + ".rlc-bearer.csv", rlcBearerSampleMs, simTime);
        rlcBearerProbe->Start();
    }
    Simulator::Schedule(Seconds(std::max(0.001, trafficStartSec - 0.001)),
                        [&delayTracker]() {
                            bool connected = Config::ConnectFailSafe(
                                "/NodeList/*/DeviceList/*/NrGnbRrc/UeMap/*/DataRadioBearerMap/*/NrRlc/TxDrop",
                                MakeCallback(&PacketDelayTracker::RecordDropWithContext, &delayTracker));
                            std::cout << "RlcTxDropTraceConnected=" << (connected ? 1 : 0)
                                      << std::endl;
                        });
    if (trafficStopSec > 0.0)
    {
        Simulator::Schedule(Seconds(trafficStopSec), [&scheduler]() {
            PrintDlBearerBufferStats(scheduler, "traffic-stop");
        });
    }
    clientApps.Start(Seconds(trafficStartSec));
    if (trafficStopSec > 0.0)
    {
        clientApps.Stop(Seconds(trafficStopSec));
        Simulator::Schedule(Seconds(trafficStopSec), [&delayTracker, &pendingAtTrafficStop]() {
            pendingAtTrafficStop = delayTracker.GetPendingBytes();
            std::cout << "PendingBytesAtTrafficStop=" << pendingAtTrafficStop << std::endl;
        });
    }

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();

    BwpState initialBwp = enableTrueBwp
                              ? static_cast<BwpState>(selectedBwpId)
                              : (enableBwpDynamic || fixedBwpMHz == 20
                                     ? BWP_NARROW_20MHZ
                                     : BWP_WIDE_100MHZ);

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
    nrEnergyHelper.SetGnbEnergyModelAttribute("ReferenceTxPowerDbm",
                                               DoubleValue(gnbEnergyReferenceTxPowerDbm));
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
    nrEnergyHelper.SetDrxModelAttribute("StartOffset", TimeValue(MilliSeconds(phaseOffsetMs)));
    nrEnergyHelper.SetDrxStartOffsets(ueCdrxStartOffsets);
    nrEnergyHelper.SetDrxInactivityTimers(ueCdrxInactivityTimers);

    energy::DeviceEnergyModelContainer gnbEnergyModels =
        nrEnergyHelper.InstallGnb(gnbNetDev, gnbEnergySources);
    energy::DeviceEnergyModelContainer ueEnergyModels =
        enableTrueBwp &&
                (enableTrueBwpSwitchScript || enablePerUeBwpSwitchScript || enablePerUeBwpAction || enableBwpDynamic ||
                 !perUeBwpMHzValues.empty())
        ? nrEnergyHelper.InstallUeAllBwps(ueNetDev, ueEnergySources, selectedBwpId)
        : nrEnergyHelper.InstallUe(ueNetDev, ueEnergySources, selectedBwpId);
    if (!ueCdrxStartOffsets.empty())
    {
        for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
        {
            Ptr<NrUeDrxModel> drx = ueNetDev.Get(ueId)->GetNode()->GetObject<NrUeDrxModel>();
            NS_ABORT_MSG_IF(!drx || drx->GetStartOffset() != ueCdrxStartOffsets[ueId],
                            "Per-UE C-DRX offset was not installed on UE " << ueId);
            std::cout << "UeCdrxOffsetMs[" << ueId
                      << "]=" << drx->GetStartOffset().GetMilliSeconds() << std::endl;
        }
    }
    if (!ueCdrxInactivityTimers.empty())
    {
        for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
        {
            Ptr<NrUeDrxModel> drx =
                ueNetDev.Get(ueId)->GetNode()->GetObject<NrUeDrxModel>();
            NS_ABORT_MSG_IF(!drx ||
                                drx->GetInactivityTimer() != ueCdrxInactivityTimers[ueId],
                            "Per-UE C-DRX inactivity timer was not installed on UE " << ueId);
            std::cout << "UeCdrxInactivityMs[" << ueId
                      << "]=" << drx->GetInactivityTimer().GetMilliSeconds() << std::endl;
        }
    }
    if (enableCdrx)
    {
        auto ueDlEligibility = [ueNetDev](uint16_t rnti, Time grantStart, Time grantEnd) {
            if (rnti == 0)
            {
                return true;
            }
            for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
            {
                Ptr<NrUeNetDevice> ue = DynamicCast<NrUeNetDevice>(ueNetDev.Get(ueId));
                if (ue && ue->GetRrc() && ue->GetRrc()->GetRnti() == rnti)
                {
                    Ptr<NrUeDrxModel> drx = ue->GetNode()->GetObject<NrUeDrxModel>();
                    NS_ABORT_MSG_IF(!drx, "C-DRX is enabled but the UE has no DRX model");
                    return drx->IsActiveFor(grantStart, grantEnd);
                }
            }
            // RNTI assignment and initial access happen before connected-mode
            // DRX applies. Unknown temporary RNTIs must remain schedulable.
            return true;
        };
        if (physicalSchedulers.empty())
        {
            scheduler->SetUeDlEligibilityCallback(ueDlEligibility);
        }
        else
        {
            for (const auto& physicalScheduler : physicalSchedulers)
            {
                physicalScheduler->SetUeDlEligibilityCallback(ueDlEligibility);
            }
        }
    }
    std::vector<Ptr<NrUeDrxModel>> ueDrxModels;
    ueDrxModels.reserve(ueNetDev.GetN());
    for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
    {
        ueDrxModels.push_back(
            ueNetDev.Get(ueId)->GetNode()->GetObject<NrUeDrxModel>());
    }
    const Time eligibilitySampleInterval =
        Seconds(0.001 / static_cast<double>(1u << numerology));
    EligibilityOverlapTracker eligibilityTracker(ueDrxModels,
                                                  &radioController,
                                                  eligibilitySampleInterval,
                                                  Seconds(simTime));
    OfficialEnergyMetrics energyMetrics(gnbEnergyModels, ueEnergyModels, ueNodes);
    Ptr<NrGnbEnergyModel> gnbEnergyModel =
        DynamicCast<NrGnbEnergyModel>(gnbEnergyModels.Get(0));
    NS_ABORT_MSG_IF(!gnbEnergyModel,
                    "The Cell-DTX controller needs the carrier-level gNB energy model");
    radioController.SetGnbEnergyModel(gnbEnergyModel);
    if (enableTrueBwp)
    {
        const double initializeEnergySec = std::max(0.0, trafficStartSec - 1e-6);
        if (!perUeBwpMHzValues.empty())
        {
            Simulator::Schedule(Seconds(initializeEnergySec),
                                &ApplyPerUeEnergyBwpScaling,
                                ueEnergyModels,
                                perUeBwpMHzValues);
        }
        else
        {
            Simulator::Schedule(Seconds(initializeEnergySec),
                                &ApplyUeEnergyBwpScaling,
                                ueEnergyModels,
                                selectedBwpId == BWP_WIDE_100MHZ ? 100u : 20u);
        }
        if (!perUeBwpMHzValues.empty())
        {
            const double switchTimeSec = std::max(0.01, trafficStartSec - 0.05);
            for (uint32_t ueId = 0; ueId < perUeBwpMHzValues.size(); ++ueId)
            {
                const uint8_t targetBwp = perUeBwpMHzValues[ueId] == 100
                                               ? BWP_WIDE_100MHZ
                                               : BWP_NARROW_20MHZ;
                if (targetBwp != selectedBwpId)
                {
                    Simulator::Schedule(Seconds(switchTimeSec),
                                        &SwitchUeToPrimaryBwp,
                                        ueNetDev,
                                        ueId,
                                        targetBwp);
                }
            }
        }
        if (enablePerUeBwpSwitchScript)
        {
            Simulator::Schedule(Seconds(perUeBwpSwitchToWideSec),
                                &SwitchUeToPrimaryBwpAndEnergy,
                                ueNetDev,
                                ueEnergyModels,
                                perUeBwpSwitchUeId,
                                static_cast<uint8_t>(BWP_WIDE_100MHZ));
            Simulator::Schedule(Seconds(perUeBwpSwitchBackToNarrowSec),
                                &SwitchUeToPrimaryBwpAndEnergy,
                                ueNetDev,
                                ueEnergyModels,
                                perUeBwpSwitchUeId,
                                static_cast<uint8_t>(BWP_NARROW_20MHZ));
        }
        if (selectedBwpId != BWP_NARROW_20MHZ)
        {
            const double switchTimeSec = std::max(0.01, trafficStartSec - 0.05);
            Simulator::Schedule(Seconds(switchTimeSec),
                                &SwitchAllUesToPrimaryBwpAndEnergy,
                                ueNetDev,
                                ueEnergyModels,
                                selectedBwpId);
        }
        if (enableTrueBwpSwitchScript)
        {
            NS_ABORT_MSG_IF(bwpSwitchToWideSec <= trafficStartSec,
                            "The first scripted BWP switch must occur after traffic starts");
            Simulator::Schedule(Seconds(bwpSwitchToWideSec),
                                &SwitchAllUesToPrimaryBwpAndEnergyWithProbe,
                                ueNetDev,
                                ueEnergyModels,
                                bwpHandoverProbe.get(),
                                tbContinuityProbe.get(),
                                static_cast<uint8_t>(BWP_WIDE_100MHZ));
            Simulator::Schedule(Seconds(bwpSwitchBackToNarrowSec),
                                &SwitchAllUesToPrimaryBwpAndEnergyWithProbe,
                                ueNetDev,
                                ueEnergyModels,
                                bwpHandoverProbe.get(),
                                tbContinuityProbe.get(),
                                static_cast<uint8_t>(BWP_NARROW_20MHZ));
        }
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
                                           controllerQueueWarnRatio, controllerQueueHighRatio,
                                           controllerBwpDwellMs,
                                           timelineCsvName,
                                           controllerWarmupMs,
                                           initialBwp,
                                           enableTrueBwp && enableBwpDynamic,
                                           enablePerUeBwpAction,
                                           ueNetDev,
                                           ueEnergyModels);
    bool dynamicControlEnabled = enableQueueAware || enableRlIpc || enableBwpDynamic ||
                                 enablePerUeBwpAction;
    radioController.Start(cellDtxActiveMs, initialBwp);
    eligibilityTracker.Start(Seconds(trafficStartSec));
    if (dynamicControlEnabled)
    {
        dynamicController.Start();
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer flowStats = monitor->GetFlowStats();
    uint64_t totalRxBytes = 0;
    uint64_t flowTxPackets = 0;
    uint64_t flowRxPackets = 0;
    uint64_t flowLostPackets = 0;
    uint64_t flowTxBytes = 0;
    uint64_t flowRxBytes = 0;
    for (auto iter = flowStats.begin(); iter != flowStats.end(); ++iter)
    {
        flowTxPackets += iter->second.txPackets;
        flowRxPackets += iter->second.rxPackets;
        flowLostPackets += iter->second.lostPackets;
        flowTxBytes += iter->second.txBytes;
        flowRxBytes += iter->second.rxBytes;
        totalRxBytes += iter->second.rxBytes;
    }

    double finalActiveMs = dynamicControlEnabled ? dynamicController.GetCurrentActiveMs() : cellDtxActiveMs;
    BwpState finalBwp = dynamicControlEnabled ? dynamicController.GetCurrentBwp() : initialBwp;
    double avgPower = energyMetrics.GetAvgGnbPowerW();
    double totalEnergy = energyMetrics.GetTotalGnbEnergyJ();
    double totalRxMbits = (totalRxBytes * 8.0) / 1e6;
    double energyEff = (totalEnergy > 0) ? (totalRxMbits / totalEnergy) : 0.0;
    double p99Latency = delayTracker.GetP99Latency();
    double p50Latency = delayTracker.GetLatencyPercentile(0.50);
    double p95Latency = delayTracker.GetLatencyPercentile(0.95);
    double qosSatisfaction = delayTracker.GetQosSatisfactionRate(qosTargetDelayMs);
    double qos20 = delayTracker.GetQosSatisfactionRate(20.0);
    double qos30 = delayTracker.GetQosSatisfactionRate(30.0);
    double qos50 = delayTracker.GetQosSatisfactionRate(50.0);
    std::string perUeP99LatencyMs = delayTracker.GetPerUeLatencyPercentile(0.99);
    std::string perUeQos20Pct = delayTracker.GetPerUeQosSatisfactionRate(20.0);
    std::string perUeQos30Pct = delayTracker.GetPerUeQosSatisfactionRate(30.0);
    std::string perUeQos50Pct = delayTracker.GetPerUeQosSatisfactionRate(50.0);
    double ueAwakePct = eligibilityTracker.GetAwakePct();
    double ueAwakeCellInactivePct = eligibilityTracker.GetAwakeCellInactivePct();
    double wasteWithinUeAwakePct = eligibilityTracker.GetWasteWithinAwakePct();
    std::string ueAwakeCellInactivePctPerUe =
        eligibilityTracker.GetAwakeCellInactivePctPerUe();
    std::string wasteWithinUeAwakePctPerUe =
        eligibilityTracker.GetWasteWithinAwakePctPerUe();

    uint64_t rlcTxDropPackets = 0;
    uint64_t rlcTxDropBytes = 0;
    for (const auto& [key, counters] : delayTracker.GetDropByBearer())
    {
        rlcTxDropPackets += counters.first;
        rlcTxDropBytes += counters.second;
    }
    const uint64_t harqNackEvents = harqDiagnostics.GetTotalNacks();
    const uint32_t harqMaxRetx = harqDiagnostics.GetMaxRetx();
    const uint64_t unattributedFlowLostPackets =
        flowLostPackets > rlcTxDropPackets ? flowLostPackets - rlcTxDropPackets : 0;

    std::cout << "OfferedTxBytes=" << delayTracker.GetTotalTxBytes()
              << ", PendingBytes=" << delayTracker.GetPendingBytes()
              << ", PendingBytesAtTrafficStop=" << pendingAtTrafficStop
              << ", AppRxBytes=" << delayTracker.GetTotalRxBytes()
              << ", TrackerDropBytes=" << delayTracker.GetTotalDropBytes()
              << ", FlowTxPackets=" << flowTxPackets
              << ", FlowRxPackets=" << flowRxPackets
              << ", FlowLostPackets=" << flowLostPackets
              << ", RlcTxDropPackets=" << rlcTxDropPackets
              << ", RlcTxDropBytes=" << rlcTxDropBytes
              << ", HarqNackEvents=" << harqNackEvents
              << ", HarqMaxRetx=" << harqMaxRetx
              << ", UnattributedFlowLostPackets=" << unattributedFlowLostPackets
              << ", FlowTxBytes=" << flowTxBytes
              << ", FlowRxBytes=" << flowRxBytes << std::endl;
    std::cout << "UeAwakePct=" << ueAwakePct
              << ", UeAwakeCellInactivePct=" << ueAwakeCellInactivePct
              << ", WasteWithinUeAwakePct=" << wasteWithinUeAwakePct
              << ", UeAwakeCellInactivePctPerUe=" << ueAwakeCellInactivePctPerUe
              << ", WasteWithinUeAwakePctPerUe=" << wasteWithinUeAwakePctPerUe
              << std::endl;
    std::cout << "PerUeP99LatencyMs=" << perUeP99LatencyMs
              << ", PerUeQos20Pct=" << perUeQos20Pct
              << ", PerUeQos30Pct=" << perUeQos30Pct
              << ", PerUeQos50Pct=" << perUeQos50Pct << std::endl;
    PrintDlBearerBufferStats(scheduler, "simulation-end");
    for (const auto& [key, counters] : delayTracker.GetDropByBearer())
    {
        std::cout << "RlcBearerDrop,Rnti=" << key.first
                  << ",Lcid=" << static_cast<uint32_t>(key.second)
                  << ",Packets=" << counters.first
                  << ",Bytes=" << counters.second << std::endl;
    }
    if (rlcBearerProbe)
    {
        rlcBearerProbe->PrintPeaks();
    }
    harqDiagnostics.Print();

    double ueAvgPowerMw = energyMetrics.GetAvgUePowerMwPerUe();
    double ueTotalEnergy = energyMetrics.GetTotalUeEnergyJ();
    double ueBatteryMah = energyMetrics.GetBatteryConsumptionMah();
    double ue3EnergyJoules = energyMetrics.GetUeEnergyJ(2);
    for (uint32_t ueId = 0; ueId < ueNetDev.GetN(); ++ueId)
    {
        std::cout << "UeEnergyJ[" << ueId << "]=" << energyMetrics.GetUeEnergyJ(ueId)
                  << std::endl;
    }

    std::cout << "EnergyModel=5G-LENA-Official-TR38.864/TR38.840"
              << ", GnbRefConfig=" << gnbEnergyRefConfig
              << ", GnbBsCategory=" << gnbEnergyBsCategory
              << ", GnbPowerUnitW=" << gnbEnergyPowerUnitW
              << ", UePowerUnitMw=" << ueEnergyPowerUnitMw
              << ", FrequencyControl="
              << (enableTrueBwpSwitchScript
                      ? "scripted-physical-BWP-v6"
                      : (enableTrueBwp && enableBwpDynamic
                             ? (enableRlIpc ? "ipc-physical-BWP-v6"
                                            : "rule-physical-BWP-v6")
                             : (enableTrueBwp ? "fixed-physical-BWP-v6-prototype"
                                              : "PRB-profile-on-100MHz-PHY")))
              << ", ActivePhysicalBwpMHz="
              << (enableTrueBwp
                      ? static_cast<uint32_t>(g_bwpResourceProfiles[finalBwp].channelBwMhz)
                      : 100u)
              << std::endl;

    ExportStatsToCsv(csvFileName, Simulator::Now().GetSeconds(), numUes, enableGating, enableQueueAware, cellDtxPeriodMs, finalActiveMs, udpPacketRateMbps, totalRxBytes, delayTracker.GetTotalDropBytes(), delayTracker.GetPendingBytes(), pendingAtTrafficStop, rlcTxDropPackets, rlcTxDropBytes, harqNackEvents, harqMaxRetx, unattributedFlowLostPackets, flowTxPackets, flowRxPackets, flowLostPackets, flowTxBytes, flowRxBytes, avgPower, totalEnergy, energyEff, p99Latency, qosSatisfaction, enableCdrx, enableDciDynamic, enableBwpDynamic, phaseOffsetMs, ueCdrxOffsetsMs, cdrxInactivityMs, ueCdrxInactivityMs, ueAvgPowerMw, ueTotalEnergy, ueBatteryMah, ue3EnergyJoules, p50Latency, p95Latency, qos20, qos30, qos50, ueAwakePct, ueAwakeCellInactivePct, wasteWithinUeAwakePct, ueAwakeCellInactivePctPerUe, wasteWithinUeAwakePctPerUe, perUeP99LatencyMs, perUeQos20Pct, perUeQos30Pct, perUeQos50Pct);

    Simulator::Destroy();
    return 0;
}
