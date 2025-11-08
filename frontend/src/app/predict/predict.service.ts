// src/app/predict/predict.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpEventType, HttpRequest } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

// Interfaces for returned JSON shape (lightweight)
export interface Top3Entry { label_idx: number; label_raw: number; prob: number; }
export interface WindowRow {
  true?: number | null;
  pred?: number | null;
  correct?: boolean;
  top3?: Top3Entry[];
}
export interface BatchResult {
  batch_index: number;
  windows: WindowRow[];
  loss?: number | null;
  acc?: string | null;
}
export interface PerFileSummary {
  file: string;
  majority_predicted?: number | null;
  majority_actual?: number | null;
  n_windows?: number;
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
  // Change these endpoints if your Flask app is served on another path/origin
  private predictEndpoint = '/gaitid/predict';
  private latestJsonEndpoint = '/gaitid/prediction_result_latest.json';

  constructor(private http: HttpClient) {}

  /**
   * Upload files to backend and emit progress updates.
   * Observable emits objects: { progress?: number, done?: boolean, response?: any }
   */
  uploadFiles(files: File[], endpoint: string = this.predictEndpoint): Observable<{ progress?: number; done?: boolean; response?: PredictionJSON | any }> {
    const form = new FormData();
    files.forEach(f => form.append('file', f, f.name));

    const req = new HttpRequest('POST', endpoint, form, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req).pipe(
      map((event: HttpEvent<any>) => {
        switch (event.type) {
          case HttpEventType.Sent:
            return { progress: 0 };
          case HttpEventType.UploadProgress:
            const percent = event.total ? Math.round(100 * (event.loaded / event.total)) : 0;
            return { progress: percent };
          case HttpEventType.Response:
            return { progress: 100, done: true, response: event.body };
          default:
            return {};
        }
      })
    );
  }

  /** Fetch latest static JSON (if Flask saves a latest JSON file) */
  fetchLatestJSON(url: string = this.latestJsonEndpoint): Observable<PredictionJSON> {
    return this.http.get<PredictionJSON>(url);
  }
}
