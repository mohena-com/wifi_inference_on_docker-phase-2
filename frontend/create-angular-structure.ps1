# Ensure src and env folders exist
New-Item -ItemType Directory -Force -Path .\src\environments | Out-Null

# environment.ts (dev)
@"
export const environment = {
  production: false,
  apiBaseUrl: 'http://localhost:5002'
};
"@ | Out-File -Encoding utf8 .\src\environments\environment.ts

# environment.prod.ts (prod)
@"
export const environment = {
  production: true,
  apiBaseUrl: '/gaitid'
};
"@ | Out-File -Encoding utf8 .\src\environments\environment.prod.ts

# polyfills.ts (minimal)
@"
// Minimal polyfills for Angular (zone.js)
import 'zone.js';
"@ | Out-File -Encoding utf8 .\src\polyfills.ts

# main.ts (bootstrap)
@"
import { enableProdMode } from '@angular/core';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';
import { AppModule } from './app/app.module';
import { environment } from './environments/environment';

if (environment.production) {
  enableProdMode();
}

platformBrowserDynamic().bootstrapModule(AppModule)
  .catch(err => console.error(err));
"@ | Out-File -Encoding utf8 .\src\main.ts

# index.html (minimal)
@"
<!doctype html>
<html lang=""en"">
<head>
  <meta charset=""utf-8"">
  <title>Wifi Inference Frontend</title>
  <meta name=""viewport"" content=""width=device-width, initial-scale=1"">
</head>
<body>
  <app-root>Loading...</app-root>
</body>
</html>
"@ | Out-File -Encoding utf8 .\src\index.html

# styles.scss (minimal)
@"
/* minimal global styles */
html, body {
  margin: 0;
  padding: 0;
  font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
  background: #f8fafc;
  color: #0f172a;
}
"@ | Out-File -Encoding utf8 .\src\styles.scss

# Create a minimal app.module.ts and app.component if missing
New-Item -ItemType Directory -Force -Path .\src\app | Out-Null

# app.module.ts
@"
import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';

@NgModule({
  declarations: [AppComponent],
  imports: [BrowserModule, AppRoutingModule],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
"@ | Out-File -Encoding utf8 .\src\app\app.module.ts

# app.component.ts
@"
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  template: '<router-outlet></router-outlet>',
})
export class AppComponent {}
"@ | Out-File -Encoding utf8 .\src\app\app.component.ts

# app-routing.module.ts (routes; keep predict lazy-loading route)
@"
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
"@ | Out-File -Encoding utf8 .\src\app\app-routing.module.ts

# tsconfig.app.json if missing
@"
{
  ""extends"": ""../tsconfig.json"",
  ""compilerOptions"": {
    ""outDir"": ""../out-tsc/app"",
    ""types"": []
  },
  ""files"": [
    ""main.ts"",
    ""polyfills.ts""
  ],
  ""include"": [
    ""src/**/*.d.ts""
  ]
}
"@ | Out-File -Encoding utf8 .\tsconfig.app.json

Write-Output "Minimal files created. Now run: npm run build -- --configuration production"
