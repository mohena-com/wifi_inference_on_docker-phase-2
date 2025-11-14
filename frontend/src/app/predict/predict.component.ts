import { Component } from '@angular/core';
import { PredictService } from './predict.service';
import { HttpEventType } from '@angular/common/http';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})
export class PredictComponent {
  selectedFile?: File;
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

    const fd = new FormData();
    fd.append('file', this.selectedFile);

    this.svc.uploadFile(fd).subscribe({
      next: (evt: any) => {
        if (evt.type === HttpEventType.UploadProgress) {
          this.progress = Math.round(100 * (evt.loaded / (evt.total || 1)));
        } else if (evt.type === HttpEventType.Response) {
          this.result = evt.body;
          this.uploading = false;
        }
      },error: (err) => {
        console.error('Upload failed', err);
        this.error = 'Upload failed — falling back to simulated result.';
        this.uploading = false;
        // fallback simulation
        this.svc.simulate(this.selectedFile!.name).subscribe(res => {
          this.result = res;
        });
      }
    });
  }

  clear() {
    this.selectedFile = null;
    this.result = null;
    this.progress = 0;
    this.error = null;
  }

  unique(arr: number[] = []): number[] {
    return Array.from(new Set(arr || []));
  }

  /** -----------------------------
   *  OVERALL SUMMARY CALCULATIONS
   * ----------------------------- */
  totalSamples(): number {
    if (!this.result?.batches) return 0;
    return this.result.batches.reduce(
      (sum: number, b: any) => sum + (b.total || 0),
      0
    );
  }

  totalCorrect(): number {
    if (!this.result?.batches) return 0;
    return this.result.batches.reduce(
      (sum: number, b: any) => sum + (b.correct || 0),
      0
    );
  }

  overallAccuracy(): number {
    const total = this.totalSamples();
    if (!total) return 0;
    return this.totalCorrect() / total;
  }

  overallLoss(): number {
    if (!this.result?.batches) return 0;

    let totalWeightedLoss = 0;
    let totalCount = 0;

    this.result.batches.forEach((b: any) => {
      if (b.loss !== undefined && b.total > 0) {
        totalWeightedLoss += b.loss * b.total;
        totalCount += b.total;
      }
    });

    if (totalCount === 0) return 0;
    return totalWeightedLoss / totalCount;
  }
}
