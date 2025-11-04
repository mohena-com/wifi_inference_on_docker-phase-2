const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const status = document.getElementById("status");
const results = document.getElementById("results");

const API_BASE = "/"; // same origin; for remote API change to full URL

uploadBtn.onclick = async () => {
  results.innerHTML = "";
  if (!fileInput.files.length) {
    alert("Choose a CSV file first");
    return;
  }
  const f = fileInput.files[0];
  status.textContent = "Uploading and running inference...";
  const form = new FormData();
  form.append("file", f);

  try {
    const resp = await fetch(API_BASE + "predict", { method: "POST", body: form });
    if (!resp.ok) {
      const txt = await resp.text();
      status.textContent = "Error: " + txt;
      return;
    }
    const j = await resp.json();
    status.textContent = `Got predictions for ${j.n_windows} windows`;
    renderResults(j);
  } catch (e) {
    status.textContent = "Request failed: " + e;
  }
};

function renderResults(res) {
  const preds = res.predictions;
  results.innerHTML = "";
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Window</th><th>Top label</th><th>Top prob</th><th>Top-3</th></tr></thead>";
  const tbody = document.createElement("tbody");
  preds.forEach((p, i) => {
    const tr = document.createElement("tr");
    const top3 = p.top.map(t => `${t.label} (${(t.prob*100).toFixed(1)}%)`).join(", ");
    tr.innerHTML = `<td>${i+1}</td><td>${p.top[0].label}</td><td>${(p.top[0].prob*100).toFixed(2)}%</td><td>${top3}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  results.appendChild(table);

  const canvas = document.createElement("canvas");
  canvas.style = "margin-top:12px; height:220px;";
  results.appendChild(canvas);
  const top_probs = preds.map(p => p.top[0].prob * 100);
  const labels = preds.map((p, i) => `W${i+1}`);
  const ctx = canvas.getContext("2d");
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Top-prob per window',
        data: top_probs
      }]
    },
    options: {
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: 'Probability (%)' } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  const dlBtn = document.createElement("button");
  dlBtn.textContent = "Download JSON results";
  dlBtn.onclick = () => {
    const blob = new Blob([JSON.stringify(res, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "predictions.json";
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  };
  results.appendChild(dlBtn);
}
