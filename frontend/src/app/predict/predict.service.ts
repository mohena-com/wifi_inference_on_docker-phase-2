import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent, HttpEventType, HttpRequest } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class PredictService {
  // use proxy in dev (ng serve --proxy-config) so this can be '/api/predict'
  private endpoint = '/gaitid/predict';

  constructor(private http: HttpClient) {}

  uploadFile(file: File): Observable<any> {
    const form = new FormData();
    form.append('file', file, file.name);

    const req = new HttpRequest('POST', this.endpoint, form, {
      reportProgress: true
    });

    return this.http.request(req).pipe(
      map((event: HttpEvent<any>) => {
        if (event.type === HttpEventType.UploadProgress) {
          const percent = Math.round(100 * (event.loaded || 0) / (event.total || 1));
          return { type: 'progress', progress: percent };
        } else if (event.type === HttpEventType.Response) {
          return { type: 'result', result: event.body };
        } else {
          return { type: 'event', event };
        }
      }),
      catchError((err) => throwError(() => err))
    );
  }

  // offline/demo fallback
  simulate(fileName: string) {
    const sample = {
      batches: [
        { accuracy: '2/8', batch: 1, correct: 2, loss: 2.942568302154541, predicted_value: [10,23,23,23,22,10,10,22], total: 8, true_value: [22,22,22,22,22,22,22,22] },
        { accuracy: '1/8', batch: 2, correct: 1, loss: 5.350837230682373, predicted_value: [5,6,10,9,3,22,10,23], total: 8, true_value: [22,22,22,22,22,22,22,22] },
        { accuracy: '1/2', batch: 3, correct: 1, loss: 0.9598658084869385, predicted_value: [10,22], total: 2, true_value: [22,22] }
      ],
      file_list: [`/tmp/uploads/${fileName}`]
    };
    return of(sample);
  }
}
