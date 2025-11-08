import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

const routes: Routes = [
  { path: '', redirectTo: 'predict', pathMatch: 'full' },
  { path: 'predict', loadChildren: () => import('./predict/predict.module').then(m => m.PredictModule) },
  { path: '**', redirectTo: 'predict' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes, { scrollPositionRestoration: 'enabled' })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
