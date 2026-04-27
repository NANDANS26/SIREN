"""
SIREN Scenario Library — real attack patterns for LLM reasoning.

Each scenario gives the LLM agent a REAL PROBLEM to reason about:
- context: natural language description of what is happening
- system_logs: simulated log lines the agent reads
- indicators_of_compromise: observable evidence to detect
- correct_action_sequence: what an expert SOC analyst would do
- severity: ground truth severity (not randomized)

The agent must READ these fields to make a decision, not just pattern-match numbers.
This mirrors how reasoning_gym gives the agent a question to answer and repl_env
gives the agent code output to reason about.

10 scenarios based on real attack patterns from MITRE ATT&CK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Scenario:
    """A real security incident scenario for LLM reasoning."""
    id: str
    name: str
    attack_type: str                        # MITRE ATT&CK category
    context: str                            # natural language description
    system_logs: List[str]                  # simulated log lines
    indicators_of_compromise: List[str]     # observable evidence
    correct_action_sequence: List[str]      # expert SOC response
    severity: int                           # ground truth 1–5
    requires_isolation: bool = False        # must isolate before patching
    task_ids: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])


# ---------------------------------------------------------------------------
# 10 real attack scenarios
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [

    Scenario(
        id="SC-001",
        name="Phishing Email with Credential Harvest",
        attack_type="phishing",
        context=(
            "A user on web-server-02 clicked a phishing link in an email purporting to be "
            "from IT support. The link redirected to a credential harvesting page. "
            "The user's session token was exfiltrated. No malware was installed. "
            "This is a low-severity incident requiring standard containment."
        ),
        system_logs=[
            "2024-01-15 09:23:11 WARN auth-service: Failed login attempt from 192.168.1.45",
            "2024-01-15 09:23:14 INFO web-server-02: Outbound HTTP GET to suspicious-domain.ru/harvest",
            "2024-01-15 09:23:15 WARN auth-service: Session token reuse detected for user jsmith",
            "2024-01-15 09:23:18 INFO firewall: Blocked outbound connection to 185.220.101.45:443",
        ],
        indicators_of_compromise=[
            "Outbound connection to suspicious-domain.ru",
            "Session token reuse from different IP",
            "Failed login attempts following phishing click",
        ],
        correct_action_sequence=["verified_patch"],
        severity=3,
        requires_isolation=False,
        task_ids=["easy"],
    ),

    Scenario(
        id="SC-002",
        name="Ransomware Encryption in Progress",
        attack_type="ransomware",
        context=(
            "db-prod-01 is exhibiting signs of active ransomware encryption. "
            "File system activity shows mass renaming of .sql and .bak files to .locked extension. "
            "The ransomware process is communicating with a C2 server. "
            "CRITICAL: The database must be isolated immediately to prevent spread to auth-service-03. "
            "Do NOT patch without isolating first — the encryption process will corrupt the patch."
        ),
        system_logs=[
            "2024-01-15 14:02:33 CRIT db-prod-01: Mass file rename detected — 847 files renamed to .locked",
            "2024-01-15 14:02:34 CRIT db-prod-01: Process svchost.exe spawning cmd.exe with vssadmin delete shadows",
            "2024-01-15 14:02:35 WARN firewall: Outbound C2 beacon to 10.0.0.99:4444 from db-prod-01",
            "2024-01-15 14:02:36 CRIT db-prod-01: CPU at 98%, disk I/O at maximum — encryption in progress",
            "2024-01-15 14:02:40 WARN auth-service-03: Unusual SMB connection attempt from db-prod-01",
        ],
        indicators_of_compromise=[
            "Mass file renaming to .locked extension",
            "vssadmin delete shadows command (destroys backups)",
            "C2 beacon on port 4444",
            "Lateral movement attempt via SMB",
        ],
        correct_action_sequence=["isolate_system", "verified_patch"],
        severity=5,
        requires_isolation=True,
        task_ids=["medium", "hard"],
    ),

    Scenario(
        id="SC-003",
        name="SQL Injection — Data Exfiltration",
        attack_type="sql_injection",
        context=(
            "auth-service-03 is under active SQL injection attack. "
            "The attacker is using UNION-based injection to extract the users table. "
            "Approximately 12,000 user records including password hashes have been exfiltrated. "
            "The attack is ongoing. Audit mode is active — all response actions are being logged "
            "for compliance review. Use verified_patch only."
        ),
        system_logs=[
            "2024-01-15 11:15:02 WARN auth-service-03: SQL error in query: UNION SELECT username,password FROM users--",
            "2024-01-15 11:15:03 WARN auth-service-03: Unusual data volume in HTTP response: 2.3MB for /api/login",
            "2024-01-15 11:15:05 INFO db-prod-01: SELECT * FROM users executed 847 times in 60 seconds",
            "2024-01-15 11:15:08 CRIT auth-service-03: Data exfiltration detected — 12,847 records sent to 203.0.113.42",
        ],
        indicators_of_compromise=[
            "UNION SELECT in HTTP parameters",
            "Abnormal response size (2.3MB for login endpoint)",
            "High-frequency SELECT on users table",
            "Large outbound data transfer to external IP",
        ],
        correct_action_sequence=["verified_patch"],
        severity=4,
        requires_isolation=False,
        task_ids=["medium"],
    ),

    Scenario(
        id="SC-004",
        name="Credential Stuffing Attack",
        attack_type="credential_stuffing",
        context=(
            "auth-service-03 is experiencing a credential stuffing attack. "
            "An attacker is using a list of 50,000 username/password pairs from a previous breach. "
            "The attack is distributed across 200 IP addresses to evade rate limiting. "
            "Current success rate: 0.3% (150 accounts compromised). "
            "Standard patch is sufficient — no isolation required."
        ),
        system_logs=[
            "2024-01-15 08:00:01 WARN auth-service-03: 847 failed logins in last 60 seconds",
            "2024-01-15 08:00:02 WARN auth-service-03: Login attempts from 200 distinct IPs",
            "2024-01-15 08:00:05 CRIT auth-service-03: 150 successful logins with known-breached credentials",
            "2024-01-15 08:00:10 INFO firewall: Rate limit triggered for /api/login endpoint",
        ],
        indicators_of_compromise=[
            "High volume of failed logins (847/min)",
            "Distributed source IPs (200 distinct)",
            "Successful logins matching breach database",
        ],
        correct_action_sequence=["verified_patch"],
        severity=4,
        requires_isolation=False,
        task_ids=["medium"],
    ),

    Scenario(
        id="SC-005",
        name="Insider Threat — Unauthorized Data Access",
        attack_type="insider_threat",
        context=(
            "auth-service-03 shows signs of insider threat activity. "
            "A privileged user account (admin_backup) is accessing sensitive customer records "
            "outside normal business hours. The account has downloaded 45GB of data in 2 hours. "
            "The user's badge shows they are not in the building. Account may be compromised. "
            "Escalate to human review before taking automated action — this requires HR involvement."
        ),
        system_logs=[
            "2024-01-15 02:14:33 WARN auth-service-03: admin_backup accessed 8,432 customer records at 02:14 AM",
            "2024-01-15 02:14:35 WARN auth-service-03: admin_backup downloading bulk export — 45GB transferred",
            "2024-01-15 02:14:40 INFO badge-system: admin_backup badge NOT scanned today",
            "2024-01-15 02:14:45 WARN auth-service-03: admin_backup login from IP 198.51.100.23 (VPN exit node)",
        ],
        indicators_of_compromise=[
            "After-hours access by privileged account",
            "Bulk data export (45GB)",
            "Physical badge not scanned (user not in building)",
            "Login from VPN exit node",
        ],
        correct_action_sequence=["escalate_human", "verified_patch"],
        severity=3,
        requires_isolation=False,
        task_ids=["medium", "hard"],
    ),

    Scenario(
        id="SC-006",
        name="DDoS — Volumetric Attack",
        attack_type="ddos",
        context=(
            "web-server-02 is under a volumetric DDoS attack. "
            "Incoming traffic has spiked to 45Gbps from a botnet of 12,000 nodes. "
            "The server is responding slowly but has not gone down. "
            "SLA credits are draining rapidly due to degraded service. "
            "Apply verified patch to activate DDoS mitigation rules."
        ),
        system_logs=[
            "2024-01-15 16:30:01 CRIT web-server-02: Incoming traffic 45Gbps — 300x normal baseline",
            "2024-01-15 16:30:02 WARN web-server-02: Response time degraded to 8.2 seconds (SLA: 2s)",
            "2024-01-15 16:30:05 INFO firewall: 12,847 distinct source IPs in last 60 seconds",
            "2024-01-15 16:30:10 WARN web-server-02: Connection table 94% full — risk of service failure",
        ],
        indicators_of_compromise=[
            "Traffic spike 300x baseline (45Gbps)",
            "Response time 4x SLA threshold",
            "12,847 distinct source IPs (botnet signature)",
            "Connection table near capacity",
        ],
        correct_action_sequence=["verified_patch"],
        severity=4,
        requires_isolation=False,
        task_ids=["medium", "hard"],
    ),

    Scenario(
        id="SC-007",
        name="Supply Chain Compromise — Malicious Package",
        attack_type="supply_chain",
        context=(
            "db-prod-01 installed a compromised npm package (event-stream v3.3.6) "
            "that contains a backdoor targeting cryptocurrency wallets. "
            "The package has been executing since the last deployment 6 hours ago. "
            "The backdoor is communicating with a C2 server. "
            "CRITICAL: Isolate immediately to prevent data exfiltration, then patch."
        ),
        system_logs=[
            "2024-01-15 10:00:01 WARN db-prod-01: Unexpected outbound connection from node process to 185.220.101.45",
            "2024-01-15 10:00:03 CRIT db-prod-01: event-stream@3.3.6 flagged in npm audit — known malicious",
            "2024-01-15 10:00:05 WARN db-prod-01: Suspicious file read: /root/.bitcoin/wallet.dat",
            "2024-01-15 10:00:08 CRIT db-prod-01: Data exfiltration attempt blocked by firewall — 847KB to C2",
        ],
        indicators_of_compromise=[
            "Malicious npm package (event-stream@3.3.6)",
            "Outbound C2 connection from node process",
            "Unauthorized access to wallet.dat",
            "Blocked exfiltration attempt",
        ],
        correct_action_sequence=["isolate_system", "verified_patch"],
        severity=5,
        requires_isolation=True,
        task_ids=["hard"],
    ),

    Scenario(
        id="SC-008",
        name="Data Exfiltration via DNS Tunneling",
        attack_type="data_exfiltration",
        context=(
            "auth-service-03 is exfiltrating data via DNS tunneling. "
            "An attacker has encoded sensitive data in DNS query subdomains "
            "to bypass firewall rules that block direct outbound connections. "
            "Approximately 2GB of data has been exfiltrated over 4 hours. "
            "Apply verified patch to block the DNS tunneling channel."
        ),
        system_logs=[
            "2024-01-15 12:00:01 WARN dns-server: Unusual DNS query volume from auth-service-03: 8,432 queries/min",
            "2024-01-15 12:00:02 WARN dns-server: Long subdomain queries detected: aGVsbG8gd29ybGQ.evil-c2.com",
            "2024-01-15 12:00:05 CRIT auth-service-03: DNS query entropy score 7.8/8.0 — tunneling signature",
            "2024-01-15 12:00:10 WARN firewall: 2.1GB outbound via DNS over last 4 hours",
        ],
        indicators_of_compromise=[
            "High DNS query volume (8,432/min)",
            "Base64-encoded subdomains",
            "High entropy DNS queries (7.8/8.0)",
            "2GB outbound via DNS protocol",
        ],
        correct_action_sequence=["verified_patch"],
        severity=4,
        requires_isolation=False,
        task_ids=["medium", "hard"],
    ),

    Scenario(
        id="SC-009",
        name="Privilege Escalation — Kernel Exploit",
        attack_type="privilege_escalation",
        context=(
            "db-prod-01 shows signs of privilege escalation via kernel exploit (CVE-2023-0386). "
            "An unprivileged process has gained root access by exploiting a vulnerability "
            "in the OverlayFS filesystem. The attacker now has full system control. "
            "CRITICAL: Isolate immediately — root access means the attacker can disable logging "
            "and spread to other systems. Do not patch without isolating first."
        ),
        system_logs=[
            "2024-01-15 13:45:01 CRIT db-prod-01: Unexpected root process spawned from www-data: /bin/bash",
            "2024-01-15 13:45:02 CRIT db-prod-01: CVE-2023-0386 exploit pattern detected in kernel logs",
            "2024-01-15 13:45:03 WARN db-prod-01: /etc/passwd modified by www-data (should be root-only)",
            "2024-01-15 13:45:05 CRIT db-prod-01: New SSH key added to /root/.ssh/authorized_keys",
            "2024-01-15 13:45:08 WARN firewall: Port scan from db-prod-01 to internal network 10.0.0.0/24",
        ],
        indicators_of_compromise=[
            "Root shell spawned from web process (www-data)",
            "CVE-2023-0386 kernel exploit signature",
            "/etc/passwd modification",
            "SSH backdoor key installed",
            "Internal network port scan",
        ],
        correct_action_sequence=["isolate_system", "verified_patch"],
        severity=5,
        requires_isolation=True,
        task_ids=["hard"],
    ),

    Scenario(
        id="SC-010",
        name="Lateral Movement — Pass-the-Hash",
        attack_type="lateral_movement",
        context=(
            "An attacker has compromised web-server-02 and is performing lateral movement "
            "using Pass-the-Hash technique. They are using stolen NTLM hashes to authenticate "
            "to db-prod-01 and auth-service-03 without knowing the plaintext passwords. "
            "Multiple systems are at risk. Isolate web-server-02 to stop the spread, "
            "then apply verified patches to all affected systems."
        ),
        system_logs=[
            "2024-01-15 15:20:01 WARN web-server-02: LSASS memory dump detected — credential harvesting",
            "2024-01-15 15:20:03 CRIT db-prod-01: Authentication from web-server-02 using NTLM hash (no password)",
            "2024-01-15 15:20:05 CRIT auth-service-03: Authentication from web-server-02 using NTLM hash",
            "2024-01-15 15:20:08 WARN siem: Pass-the-Hash pattern detected across 3 systems",
            "2024-01-15 15:20:10 CRIT db-prod-01: Attacker executing commands via WMI from web-server-02",
        ],
        indicators_of_compromise=[
            "LSASS memory dump (credential harvesting)",
            "NTLM hash authentication without password",
            "WMI remote execution",
            "Lateral movement across 3 systems",
        ],
        correct_action_sequence=["isolate_system", "verified_patch", "verified_patch"],
        severity=5,
        requires_isolation=True,
        task_ids=["hard"],
    ),
]

# Index by id and by attack_type for fast lookup
SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}
SCENARIOS_BY_TYPE = {}
for _s in SCENARIOS:
    SCENARIOS_BY_TYPE.setdefault(_s.attack_type, []).append(_s)

# Scenarios suitable for each task difficulty
EASY_SCENARIOS = [s for s in SCENARIOS if "easy" in s.task_ids]
MEDIUM_SCENARIOS = [s for s in SCENARIOS if "medium" in s.task_ids]
HARD_SCENARIOS = [s for s in SCENARIOS if "hard" in s.task_ids]
