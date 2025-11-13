// src/app/predict/predict.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictComponent } from './predict.component';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  declarations: [PredictComponent],
  imports: [CommonModule, FormsModule, HttpClientModule],
  exports: [PredictComponent]   // export so router/AppModule can reference it
})
export class PredictModule {}
