export enum View {
  DASHBOARD = 'DASHBOARD',
  NEWS_SCANNER = 'NEWS_SCANNER',
  CYBER_MONITOR = 'CYBER_MONITOR',
  EMAIL_SCANNER = 'EMAIL_SCANNER',
  PROFILE = 'PROFILE',
}

export enum ThreatLevel {
  SAFE = 'Safe',
  LOW = 'Low',
  MEDIUM = 'Medium',
  HIGH = 'High',
  CRITICAL = 'Critical',
}

export interface ScoreComponent {
  key: 'evidence' | 'fact_check' | 'language' | 'prior' | string;
  label: string;
  score: number | null;
  weight: number;
  share: number;
  detail: string;
}

export interface EvidencePassage {
  title: string;
  url: string | null;
  domain: string | null;
  stance: 'entail' | 'contradict' | 'neutral';
  confidence: number;
  reliability: number;
  snippet: string;
}

export interface TopicCard {
  title: string;
  url: string | null;
  image: string | null;
  summary: string;
}

export interface CheckedClaim {
  text: string;
  label: string;
  probability: number; // P(true | evidence), 0..1
  certainty: number; // how much evidence stands behind it, 0..1
  support: number;
  refute: number;
  finding?: string;
  explore?: string[];
  topics?: TopicCard[];
  evidence: EvidencePassage[];
}

export interface NewsAnalysisResult {
  credibilityScore: number; // 0 to 100
  verdict: string; // "Credible", "Disputed", "Unverified", "Misleading", "Fake News"
  summary: string;
  reasoning: string;
  sources: string[];
  // How the score was built. Optional so older API responses still render.
  confidence?: number;
  confidence_label?: string;
  breakdown?: ScoreComponent[];
  claims?: CheckedClaim[];
  evidence_sources?: Record<string, string>;
  related?: { url: string; title: string; page_age?: string }[];
  depth?: string;
  score_source?: string;
}

export interface NetworkLog {
  id: string;
  timestamp: string;
  sourceIP: string;
  destIP: string;
  protocol: 'TCP' | 'UDP' | 'ICMP' | 'HTTP';
  packetSize: number;
  flags: string;
  // Sensor/interface the flow was captured on (e.g. "zeek/eth0"), when known.
  // Optional: browser-generated demo logs don't carry it.
  sensor?: string;
}

export interface CyberAnalysisResult {
  threatLevel: ThreatLevel;
  anomaliesDetected: string[]; // List of Log IDs
  analysisReport: string;
  recommendedAction: string;
}

export interface HistoryItem {
  id: string;
  date: string;
  type: 'NEWS' | 'CYBER';
  summary: string;
  result: string; // e.g., "Critical" or "85% Credible"
}

export interface User {
  id: string;
  name: string;
  email: string;
  history: HistoryItem[];
}
