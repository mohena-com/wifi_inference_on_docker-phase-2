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
    this.selectedFile = undefined;
    this.result = null;
    this.progress = 0;
    this.error = '';
  }

  unique_old(arr: number[] = []): number[] {
    return Array.from(new Set(arr || []));
  }

  unique(arr: Array<number | string> = []): Array<number | string> {
    const seen = new Set<string>();
    const out: Array<number | string> = [];
    for (const v of arr) {
      const key = String(v);
      if (!seen.has(key)) {
        seen.add(key);
        out.push(v);
      }
    }
    return out;
  }             
  /** -----------------------------
   *  OVERALL SUMMARY CALCULATIONS
   * ----------------------------- */
  totalSamples_old(): number {
    if (!this.result?.batches) return 0;
    return this.result.batches.reduce(
      (sum: number, b: any) => sum + (b.total || 0),
      0
    );
  }
  totalSamples(): number {
    if (!this.result || !Array.isArray(this.result.batches)) {
      return 0;
    }
    return this.result.batches.reduce((sum: number, b: any) => {
      if (typeof b.total === 'number') {
        return sum + b.total;
      }
      if (Array.isArray(b.predicted_value)) {
        return sum + b.predicted_value.length;
      }
      return sum;
    }, 0);
  }
    /** For collapsible Raw JSON panel */
  showRawJson = false;

  toggleRawJson(): void {
    this.showRawJson = !this.showRawJson;
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

  // Map raw class IDs → readable labels
  labelMap: { [key: number]: string } = {
  0: 'Class 0',
  1: 'Class 1',
  2: 'Class 2',
  3: 'Class 3',
  4: 'Class 4',
  5: 'Class 5',
  6: 'Class 6',
  7: 'Class 7',
  8: 'Class 8',
  9: 'Class 9',
  10: 'Class 10',
  11: 'Class 11',
  12: 'Class 12',
  13: 'Class 13',
  14: 'Class 14',
  15: 'Class 15',
  16: 'Class 16',
  17: 'Class 17',
  18: 'Class 18',
  19: 'Class 19',
  20: 'Class 20',
  21: 'Class 21',
  22: 'Class 22',
  23: 'Class 23',
  24: 'Class 24',
  25: 'Class 25',
  26: 'Class 26',
  27: 'Class 27',
  28: 'Class 28',
  29: 'Class 29',
  30: 'Class 30'
};

  formatLabel(v: number | string): string {
    const id = Number(v);
    return this.labelMap[id] ?? `Class ${id}`;
  }

  isCorrect(batch: any, index: number): boolean {
    if (!batch || !batch.true_value || !batch.predicted_value) {
      return false;
    }
    return batch.true_value[index] === batch.predicted_value[index];
  }
}


