import { Component, OnInit } from '@angular/core';
import { PredictService, PredictionJSON, BatchResult, WindowRow, PerFileSummary } from './predict.service';

// Charts
import { ChartConfiguration, ChartOptions } from 'chart.js';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})
export class PredictComponent implements OnInit {
  data: PredictionJSON | null = null;
  loading = false;
  error: string | null = null;

  // chart data for per-file majority summary
  chartLabels: string[] = [];
  chartDataPred: number[] = [];
  chartDataTrue: number[] = [];

  // Chart configuration
  public barChartOptions: ChartOptions = {
    responsive: true,
  };
  public barChartData: ChartConfiguration<'bar'>['data'] = {
    labels: [],
    datasets: [
      { data: [], label: 'Majority Predicted' },
      { data: [], label: 'Majority Actual' }
    ]
  };

  constructor(private svc: PredictService) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.error = null;

    // either call predict endpoint (POST with files) or fetch static JSON
    // For demo: use static file endpoint (adjust to your server)
    this.svc.fetchPredictionFile('/prediction_result_latest.json').subscribe({
      next: (d) => { this.data = d; this.prepareChart(); this.loading = false; },
      error: (err) => { this.error = err.message || 'Failed to fetch'; this.loading = false; }
    });
  }

  // Helpers
  iconFor(correct?: boolean): string { return correct ? '✅' : '❌'; }
  colorFor(correct?: boolean): string { return correct ? '#16a34a' : '#dc2626'; }

  // Expand/Collapse state
  expandedBatches = new Set<number>();
  toggleBatch(idx: number) {
    if (this.expandedBatches.has(idx)) this.expandedBatches.delete(idx); else this.expandedBatches.add(idx);
  }
  isExpanded(idx: number) { return this.expandedBatches.has(idx); }

  // Prepare chart from per_file_summary
  prepareChart() {
    if (!this.data || !this.data.per_file_summary) return;
    const pf = this.data.per_file_summary;
    this.barChartData.labels = pf.map(x => x.file);
    this.barChartData.datasets[0].data = pf.map(x => x.majority_predicted ?? 0);
    this.barChartData.datasets[1].data = pf.map(x => x.majority_actual ?? 0);
  }

  // Utility: format prob as percent
  fmtProb(p: number) { return (p * 100).toFixed(1) + '%'; }
}
