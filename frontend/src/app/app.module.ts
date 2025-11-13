// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { AppComponent } from './app.component';

// Import routing and feature module(s)
import { AppRoutingModule } from './app-routing.module';
import { PredictModule } from './predict/predict.module'; // assume predict.module.ts exists

@NgModule({
  declarations: [
    AppComponent,
    // DO NOT declare PredictComponent here if it is declared in PredictModule
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule,
    BrowserAnimationsModule,
    AppRoutingModule,
    PredictModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
