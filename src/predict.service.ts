import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Top3Entry { label_idx: number; label_raw: number; prob: number; }
export interface WindowRow {
  true: number | null;
  pred: number | null;
  correct: boolean;
  top3?: Top3Entry[];
  probs?: number[];
}
export interface BatchResult {
  batch_index: number;
  csi_shape?: number[];
  meta_shape?: number[];
  outputs_shape?: number[];
  windows: WindowRow[];
  loss?: number;
  acc?: string;
}
export interface PerFileSummary {
  file: string;
  majority_predicted: number | null;
  majority_actual: number | null;
  n_windows: number;
}
export interface PredictionJSON {
  batches: BatchResult[];
  summary: {
    total_windows: number;
    total_correct: number;
    total_incorrect: number;
    notes?: string;
  };
  per_file_summary?: PerFileSummary[];
}

@Injectable({
  providedIn: 'root'
})
export class PredictService {
  constructor(private http: HttpClient) {}

  // change URL to your Flask predict endpoint
  fetchPrediction(endpoint: string = '/gaitid/predict'): Observable<PredictionJSON> {
    return this.http.post<PredictionJSON>(endpoint, new FormData()); // if you want GET a static json use this.http.get(...)
  }

  // Or fetch static JSON file:
  fetchPredictionFile(url: string = '/uploads/prediction_result_latest.json') {
    return this.http.get<PredictionJSON>(url);
  }
}
