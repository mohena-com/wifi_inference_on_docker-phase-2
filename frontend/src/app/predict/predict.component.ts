// src/app/predict/predict.component.ts
import { Component } from '@angular/core';
import { PredictService } from './predict.service';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})
export class PredictComponent {
  selectedFile?: File;
  selectedFiles?: FileList;
  uploading = false;
  progress = 0;
  result: any = null;
  error = '';

  constructor(private svc: PredictService) {}

  onFileChange(e: any) {
    const f = e.target.files?.[0];
    if (f) {
      this.selectedFile = f;
      this.result = null;
      this.progress = 0;
      this.error = '';
    }
  }

  upload() {
    if (!this.selectedFile) {
      this.error = 'Please select a file first.';
      return;
    }
    this.uploading = true;
    this.progress = 0;
    this.error = '';

    this.svc.uploadFile(this.selectedFile).subscribe({
      next: (evt) => {
        if (evt.type === 'progress') {
          this.progress = evt.progress;
        } else if (evt.type === 'result') {
          this.result = evt.result;
          this.uploading = false;
        }
      },
      error: (err) => {
        console.error('Upload failed', err);
        this.svc.simulate(this.selectedFile!.name).subscribe(res => {
          this.result = res;
          this.uploading = false;
          this.error = 'Server unreachable — using simulated response.';
        });
      }
    });
  }

  clear() {
    this.selectedFile = undefined;
    this.result = null;
    this.progress = 0;
    this.error = '';
  }

  unique(arr: number[] = []) {
    return Array.from(new Set(arr));
  }

  // NEW: compute total samples across batches
  totalSamples(): number {
    if (!this.result || !this.result.batches) return 0;
    return this.result.batches.reduce((acc: number, b: any) => acc + (b.total || 0), 0);
  }
}
