// src/app/app.module.ts
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';     // if you use HttpClient
import { FormsModule } from '@angular/forms';                // if you use template forms
import { BrowserAnimationsModule } from '@angular/platform-browser/animations'; // if Material used

import { AppComponent } from './app.component';
import { PredictComponent } from './predict/predict.component';
import { AppRoutingModule } from './app-routing.module';

@NgModule({
  declarations: [
    AppComponent,
    PredictComponent
  ],
  imports: [
    BrowserModule,            // provides CommonModule & built-in pipes (number, json, etc.)
    HttpClientModule,         // provide HttpClient
    FormsModule,
    BrowserAnimationsModule,
    AppRoutingModule      // routing module
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
