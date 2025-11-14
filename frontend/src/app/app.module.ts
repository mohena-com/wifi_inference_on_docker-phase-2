// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';

// Import the feature module (which now includes CommonModule and Material)
import { PredictModule } from './predict/predict.module';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,   // required for angular material
    HttpClientModule,
    AppRoutingModule,
    PredictModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
