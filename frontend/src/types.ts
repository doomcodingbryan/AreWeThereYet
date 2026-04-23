export interface CountryMetadata {
  region: string;
  quality_of_life_index: string;
  cost_of_living_index: string;
  safety_index: string;
  health_care_index: string;
  climate_index: string;
  official_languages: string;
  english_official: string;
  gdp_per_capita_usd: string;
  skilled_worker_visa: string;
  visa_name: string;
}

export interface LatentDimension {
  dimension: number;
  label: string;
  contribution: number;
}

export interface CountryResult {
  country: string;
  score: number;
  latent_dimensions?: LatentDimension[];
  metadata: CountryMetadata;
}
