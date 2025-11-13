// predict/predict.module.ts
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictComponent } from './predict.component';
import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [PredictComponent],
  imports: [CommonModule, FormsModule],
  exports: [PredictComponent]   // <-- export if used by AppModule routes or other modules
})
export class PredictModule {}
