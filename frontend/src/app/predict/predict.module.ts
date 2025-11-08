// src/app/predict/predict.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Routes } from '@angular/router';
import { HttpClientModule } from '@angular/common/http';
import { PredictComponent } from './predict.component';
import { NgChartsModule } from 'ng2-charts';

const routes: Routes = [
  { path: '', component: PredictComponent }
];

@NgModule({
  declarations: [
    PredictComponent
  ],
  imports: [
    CommonModule,
    HttpClientModule,
    NgChartsModule,
    RouterModule.forChild(routes)
  ]
})
export class PredictModule { }
