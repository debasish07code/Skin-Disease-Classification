/* ============================================================
   DermAI — main.js
   Handles drag-and-drop, file selection, API call, and results
   ============================================================ */

"use strict";

// ── Element refs ──────────────────────────────────────────────
const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const browseBtn       = document.getElementById("browse-btn");
const previewContainer= document.getElementById("preview-container");
const previewImg      = document.getElementById("preview-img");
const previewName     = document.getElementById("preview-name");
const clearBtn        = document.getElementById("clear-btn");
const analyzeBtn      = document.getElementById("analyze-btn");
const btnText         = analyzeBtn.querySelector(".btn-text");
const btnLoader       = analyzeBtn.querySelector(".btn-loader");

const resultsSection  = document.getElementById("results-section");
const errorCard       = document.getElementById("error-card");
const errorMsg        = document.getElementById("error-msg");
const resultCard      = document.getElementById("result-card");

const resultImg       = document.getElementById("result-img");
const diseaseIcon     = document.getElementById("disease-icon");
const resultClass     = document.getElementById("result-class");
const resultConf      = document.getElementById("result-confidence");
const severityBadge   = document.getElementById("severity-badge");
const probBars        = document.getElementById("prob-bars");
const infoClassName   = document.getElementById("info-class-name");
const infoDescription = document.getElementById("info-description");
const symptomsList    = document.getElementById("symptoms-list");
const againBtn        = document.getElementById("again-btn");

let selectedFile = null;

// ── File selection ────────────────────────────────────────────

browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", (e) => {
  if (e.target !== browseBtn) fileInput.click();
});
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// ── Drag & Drop ───────────────────────────────────────────────

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "dragend", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"));
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files[0];
  if (file) handleFile(file);
});

// ── Handle file ───────────────────────────────────────────────

function handleFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported file type. Please upload a JPG, PNG, BMP, or WEBP image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("File too large. Maximum size is 10 MB.");
    return;
  }

  selectedFile = file;
  hideResults();

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewContainer.classList.remove("hidden");
    dropZone.classList.add("hidden");
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── Clear ─────────────────────────────────────────────────────

clearBtn.addEventListener("click", resetUpload);
againBtn.addEventListener("click", resetUpload);

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewName.textContent = "";
  previewContainer.classList.add("hidden");
  dropZone.classList.remove("hidden");
  analyzeBtn.disabled = true;
  hideResults();
}

// ── Analyze ───────────────────────────────────────────────────

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // Loading state
  btnText.classList.add("hidden");
  btnLoader.classList.remove("hidden");
  analyzeBtn.disabled = true;
  hideResults();

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Network error — please check your connection and try again.");
  } finally {
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
});

// ── Render results ────────────────────────────────────────────

function renderResults(data) {
  const { predicted_class, confidence, probabilities, image_data, disease_info } = data;

  // Image
  resultImg.src = image_data;
  resultImg.alt = `Analyzed skin image — ${predicted_class}`;

  // Disease icon & name
  diseaseIcon.textContent = disease_info.icon || "🔬";
  resultClass.textContent = predicted_class;
  resultClass.style.color = disease_info.color || "#fff";

  // Confidence
  resultConf.textContent = `${confidence.toFixed(2)}%`;

  // Severity badge
  const sev = disease_info.severity || "";
  severityBadge.textContent = sev;
  if (sev.toLowerCase().includes("critical")) {
    severityBadge.style.cssText = "color:#FF6B6B;border-color:rgba(255,107,107,0.4);background:rgba(255,107,107,0.08)";
  } else if (sev.toLowerCase().includes("high")) {
    severityBadge.style.cssText = "color:#FF9F43;border-color:rgba(255,159,67,0.4);background:rgba(255,159,67,0.08)";
  } else if (sev.toLowerCase().includes("benign")) {
    severityBadge.style.cssText = "color:#00C9A7;border-color:rgba(0,201,167,0.4);background:rgba(0,201,167,0.08)";
  } else {
    severityBadge.style.cssText = "color:#8892A0;border-color:rgba(136,146,160,0.3);background:rgba(136,146,160,0.06)";
  }

  // Probability bars
  probBars.innerHTML = "";
  probabilities.forEach((item, idx) => {
    const isTop = idx === 0;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-row-header">
        <span class="prob-class">${item.class}</span>
        <span class="prob-value">${item.prob.toFixed(2)}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isTop ? "top" : "other"}" data-width="${item.prob}" style="width:0%"></div>
      </div>
    `;
    probBars.appendChild(row);
  });

  // Animate bars after a brief delay
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll(".prob-bar-fill").forEach((bar) => {
        bar.style.width = `${bar.dataset.width}%`;
      });
    });
  });

  // Disease info
  infoClassName.textContent = predicted_class;
  infoDescription.textContent = disease_info.description || "";
  symptomsList.innerHTML = (disease_info.symptoms || [])
    .map((s) => `<li>${s}</li>`)
    .join("");

  // Show results
  resultsSection.classList.remove("hidden");
  errorCard.classList.add("hidden");
  resultCard.classList.remove("hidden");
  resultCard.classList.add("animate-in");

  // Scroll to results
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

// ── Error ─────────────────────────────────────────────────────

function showError(msg) {
  errorMsg.textContent = msg;
  resultsSection.classList.remove("hidden");
  errorCard.classList.remove("hidden");
  resultCard.classList.add("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function hideResults() {
  resultsSection.classList.add("hidden");
  errorCard.classList.add("hidden");
  resultCard.classList.add("hidden");
}
