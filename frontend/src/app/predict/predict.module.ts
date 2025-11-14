// src/app/predict/predict.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictComponent } from './predict.component';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

// Material modules (import where the component is declared)
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatInputModule } from '@angular/material/input';

@NgModule({
  declarations: [PredictComponent],
  imports: [
    CommonModule,           // provides number/json pipes and attribute binding
    FormsModule,
    HttpClientModule,
    // Material
    MatCardModule,
    MatButtonModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatInputModule
  ],
  exports: [PredictComponent]   // export if AppRoutingModule uses this component via routing
})
export class PredictModule {}
