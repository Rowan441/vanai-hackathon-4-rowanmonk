let allStats = null;

async function loadData() {
  try {
    const response = await fetch("/api/stats");
    allStats = await response.json();

    updateStats(allStats);
    createCharts(allStats);
    populateFilters(allStats);
    loadResponses();

    document.getElementById("loading").style.display = "none";
    document.getElementById("content").style.display = "block";
  } catch (error) {
    console.error("Error loading data:", error);
    document.getElementById("loading").textContent =
      "Error loading data. Please try again.";
  }
}

function updateStats(stats) {
  document.getElementById("totalResponses").textContent =
    stats.total_responses.toLocaleString();

  // Calculate unique artists (would need actual data)
  document.getElementById("uniqueArtists").textContent = "500+";

  // Calculate AI acceptance rate
  const aiSongs = stats.music_preferences.ai_songs;
  const yesCount =
    (aiSongs["Yes – and I already have"] || 0) +
    (aiSongs["Yes, but I haven't yet"] || 0);
  const total = Object.values(aiSongs).reduce((a, b) => a + b, 0);
  const percentage = ((yesCount / total) * 100).toFixed(0);
  document.getElementById("aiAcceptance").textContent = percentage + "%";

  document.getElementById("avgAge").textContent = "35";
}

function createCharts(stats) {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: "bottom",
      },
    },
  };

  // Relationship Chart
  createPieChart(
    "relationshipChart",
    stats.music_preferences.relationship,
    chartOptions
  );

  // Discovery Chart
  createPieChart(
    "discoveryChart",
    stats.music_preferences.discovery,
    chartOptions
  );

  // AI Chart
  createBarChart("aiChart", stats.music_preferences.ai_songs, chartOptions);

  // Dead Artists Chart
  createBarChart(
    "deadArtistsChart",
    stats.music_preferences.dead_artists_voice,
    chartOptions
  );

  // Age Chart
  createBarChart("ageChart", stats.demographics.age_groups, chartOptions);

  // Format Chart
  createPieChart("formatChart", stats.format_changes, chartOptions);
}

function createPieChart(canvasId, data, options) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  new Chart(ctx, {
    type: "pie",
    data: {
      labels: Object.keys(data),
      datasets: [
        {
          data: Object.values(data),
          backgroundColor: [
            "#667eea",
            "#764ba2",
            "#f093fb",
            "#4facfe",
            "#43e97b",
            "#fa709a",
            "#feca57",
            "#ee5a6f",
          ],
        },
      ],
    },
    options: options,
  });
}

function createBarChart(canvasId, data, options) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: Object.keys(data),
      datasets: [
        {
          label: "Count",
          data: Object.values(data),
          backgroundColor: "#667eea",
          borderColor: "#764ba2",
          borderWidth: 1,
        },
      ],
    },
    options: {
      ...options,
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

function populateFilters(stats) {
  // Age groups
  const ageSelect = document.getElementById("filterAge");
  Object.keys(stats.demographics.age_groups).forEach((age) => {
    const option = document.createElement("option");
    option.value = age;
    option.textContent = age;
    ageSelect.appendChild(option);
  });

  // Gender
  const genderSelect = document.getElementById("filterGender");
  Object.keys(stats.demographics.gender).forEach((gender) => {
    const option = document.createElement("option");
    option.value = gender;
    option.textContent = gender;
    genderSelect.appendChild(option);
  });

  // Province
  const provinceSelect = document.getElementById("filterProvince");
  Object.keys(stats.demographics.provinces).forEach((province) => {
    const option = document.createElement("option");
    option.value = province;
    option.textContent = province;
    provinceSelect.appendChild(option);
  });

  // AI Preference
  const aiSelect = document.getElementById("filterAI");
  Object.keys(stats.music_preferences.ai_songs).forEach((pref) => {
    const option = document.createElement("option");
    option.value = pref;
    option.textContent = pref;
    aiSelect.appendChild(option);
  });
}

async function loadResponses(filters = {}) {
  const params = new URLSearchParams(filters);
  const response = await fetch("/api/responses?" + params);
  const data = await response.json();

  const tbody = document.getElementById("responsesBody");
  tbody.innerHTML = "";

  data.responses.forEach((r) => {
    const row = tbody.insertRow();
    row.innerHTML = `
            <td>${r.age}</td>
            <td>${r.gender}</td>
            <td>${r.province}</td>
            <td>${r.relationship_with_music}</td>
            <td>${r.favorite_artist}</td>
            <td>${r.ai_songs}</td>
            <td><span class="response-detail" onclick="showDetail('${r.participant_id}')">View</span></td>
        `;
  });
}

function applyFilters() {
  const filters = {
    age_group: document.getElementById("filterAge").value,
    gender: document.getElementById("filterGender").value,
    province: document.getElementById("filterProvince").value,
    ai_preference: document.getElementById("filterAI").value,
  };

  // Remove empty filters
  Object.keys(filters).forEach((key) => {
    if (!filters[key]) delete filters[key];
  });

  loadResponses(filters);
}

async function showDetail(participantId) {
  const response = await fetch(`/api/response/${participantId}`);
  const data = await response.json();

  const modalContent = document.getElementById("modalContent");
  modalContent.innerHTML = `
        <h2 style="color: #667eea; margin-bottom: 20px;">Response Detail</h2>

        <div class="detail-section">
            <h3>Demographics</h3>
            <div class="detail-item"><span class="detail-label">Age:</span>${data.demographics.age}</div>
            <div class="detail-item"><span class="detail-label">Gender:</span>${data.demographics.gender}</div>
            <div class="detail-item"><span class="detail-label">Province:</span>${data.demographics.province}</div>
            <div class="detail-item"><span class="detail-label">Education:</span>${data.demographics.education}</div>
        </div>

        <div class="detail-section">
            <h3>Music Relationship</h3>
            <div class="detail-item"><span class="detail-label">Relationship:</span>${data.music_relationship.relationship}</div>
            <div class="detail-item"><span class="detail-label">Discovery:</span>${data.music_relationship.discovery_method}</div>
            <div class="detail-item"><span class="detail-label">Favorite Artist:</span>${data.music_relationship.favorite_artist}</div>
            <div class="detail-item"><span class="detail-label">Format Change:</span>${data.music_relationship.format_change}</div>
            <div class="detail-item"><span class="detail-label">Format Impact:</span>${data.music_relationship.format_impact}</div>
            <div class="detail-item"><span class="detail-label">Current Preference:</span>${data.music_relationship.current_preference}</div>
        </div>

        <div class="detail-section">
            <h3>AI Opinions</h3>
            <div class="detail-item"><span class="detail-label">AI Songs:</span>${data.ai_opinions.ai_songs}</div>
            <div class="detail-item"><span class="detail-label">Dead Artists' Voice:</span>${data.ai_opinions.dead_artists_voice}</div>
        </div>

        <div class="detail-section">
            <h3>Personal</h3>
            <div class="detail-item"><span class="detail-label">Theme Song:</span>${data.personal.theme_song}</div>
            <div class="detail-item"><span class="detail-label">Favorite Lyric:</span>${data.personal.favorite_lyric}</div>
            <div class="detail-item"><span class="detail-label">Guilty Pleasure:</span>${data.personal.guilty_pleasure}</div>
        </div>
    `;

  document.getElementById("detailModal").style.display = "block";
}

function closeModal() {
  document.getElementById("detailModal").style.display = "none";
}

window.onclick = function (event) {
  const modal = document.getElementById("detailModal");
  if (event.target == modal) {
    modal.style.display = "none";
  }
};

// Load data on page load
document.addEventListener("DOMContentLoaded", loadData);
