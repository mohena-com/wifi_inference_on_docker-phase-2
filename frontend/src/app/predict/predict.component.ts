// src/app/predict/predict.component.ts
import { Component, OnDestroy } from '@angular/core';
import { PredictService, PredictionJSON, BatchResult, WindowRow } from './predict.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})
export class PredictComponent implements OnDestroy {
  files: File[] = [];
  uploadProgress = 0;
  uploading = false;
  uploadSub?: Subscription;
  result: PredictionJSON | null = null;
  latestJsonError: string | null = null;
  uploadError: string | null = null;

  // UI state for expanding batches
  expandedBatches = new Set<number>();

  constructor(private svc: PredictService) {}

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;
    this.files = Array.from(input.files);
    this.uploadError = null;
  }

  startUpload() {
    if (!this.files || this.files.length === 0) {
      this.uploadError = 'Please select at least one CSV file.';
      return;
    }

    this.uploading = true;
    this.uploadProgress = 0;
    this.uploadError = null;
    this.result = null;

    this.uploadSub = this.svc.uploadFiles(this.files).subscribe({
      next: (evt) => {
        if (evt.progress != null) {
          this.uploadProgress = evt.progress;
        }
        if (evt.done) {
          this.uploading = false;
          this.uploadProgress = 100;
          this.result = evt.response as PredictionJSON;
        }
      },
      error: (err) => {
        this.uploading = false;
        this.uploadError = (err?.message) || 'Upload failed (server error)';
      }
    });
  }

  cancelUpload() {
    if (this.uploadSub) {
      this.uploadSub.unsubscribe();
      this.uploadSub = undefined;
    }
    this.uploading = false;
    this.uploadProgress = 0;
  }

  // load latest saved JSON (if needed)
  loadLatest() {
    this.latestJsonError = null;
    this.svc.fetchLatestJSON().subscribe({
      next: (d) => { this.result = d; },
      error: (err) => { this.latestJsonError = err?.message || 'Failed to fetch latest JSON'; }
    });
  }

  toggleBatch(batchIndex: number) {
    if (this.expandedBatches.has(batchIndex)) this.expandedBatches.delete(batchIndex);
    else this.expandedBatches.add(batchIndex);
  }
  isExpanded(batchIndex: number) {
    return this.expandedBatches.has(batchIndex);
  }

  // helpers for UI
  iconFor(correct?: boolean) { return correct ? '✅' : '❌'; }
  colorFor(correct?: boolean) { return correct ? 'var(--ok)' : 'var(--fail)'; }
  fmtProb(p: number) { return (p * 100).toFixed(1) + '%'; }

  ngOnDestroy(): void {
    this.cancelUpload();
  }
}
