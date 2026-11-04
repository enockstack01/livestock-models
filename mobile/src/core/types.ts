// Shapes of the two server APIs (see services/livestock_vision/README.md and src/livestock_llm) and of the offline advice pack.
export type ApiDetection = { class: string; confidence: number; box: number[] };

export type DetectResponse = {
  diagnosis_id: string; model_version: string; species: string; detections: ApiDetection[]; summary: Record<string, number>;
  flags: string[]; annotated_image_url: string; image: { width: number; height: number }; latency_ms: number;
  offline_comparison?: { agrees: boolean; only_online: string[]; only_offline: string[] };
};

export type Treatment = { supportive_care: string[]; medicines: string[]; requires_vet: boolean; withdrawal_periods: string[] };
export type Report = {
  summary: string; possible_conditions: string[]; severity: 'low' | 'moderate' | 'high' | 'urgent'; notifiable: boolean; zoonotic: boolean;
  immediate_actions: string[]; treatment: Treatment; prevention: string[]; follow_up_days: number | null; disclaimer?: string | null;
};
export type ReportResponse = {
  diagnosis_id: string; model_version: string; language: string; source: 'llm' | 'advice_pack'; degraded: boolean; report: Report;
  safety_adjustments: string[];
};

export type AdviceEntry = {
  label: string; aliases: string[]; title: string; species: string; notifiable: boolean; zoonotic?: boolean;
  urgency: 'none' | 'soon' | 'urgent' | 'emergency'; signs: string[]; do_now: string[]; prevention: string[]; see_vet: string;
};
export type AdvicePack = { data_version: number; reviewed_by_veterinarian: boolean; disclaimer: string; diseases: AdviceEntry[] };

export type Animal = { weight_kg?: number; age_months?: number; lactating?: boolean; pregnant?: boolean };
export type Language = 'en' | 'fr' | 'rw';
