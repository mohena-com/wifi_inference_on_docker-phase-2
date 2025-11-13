import { HttpClientModule } from '@angular/common/http';
import { PredictComponent } from './predict/predict.component';

@NgModule({
  declarations: [
    AppComponent,
    PredictComponent,
    // ...
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    // ...
  ],
  bootstrap: [AppComponent]
})
export class AppModule {}
