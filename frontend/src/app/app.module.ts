// app.module.ts (edit)
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

// App components
import { AppComponent } from './app.component';

// Import the feature module instead of declaring the component twice
import { PredictModule } from './predict/predict.module';

@NgModule({
  declarations: [
    AppComponent,
    // <-- DO NOT declare PredictComponent here if it's declared in PredictModule
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    BrowserAnimationsModule,
    PredictModule,   // <-- import the feature module
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule {}
