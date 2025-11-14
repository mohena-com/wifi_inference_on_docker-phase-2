// src/app/predict/predict.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpRequest } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PredictService {
  private endpoint = '/gaitid/predict'; // use proxy or full URL

  constructor(private http: HttpClient) {}

  // Accept FormData
  uploadFile(form: FormData): Observable<HttpEvent<any>> {
    const req = new HttpRequest('POST', this.endpoint, form, {
      reportProgress: true,
      responseType: 'json'
    });
    return this.http.request(req);
  }

  // Simulate fallback (optional)
  simulate(fileName: string) {
    const sample = { /* ...same sample as before...*/ };
    return new Observable((sub) => { sub.next(sample); sub.complete(); });
  }
}
