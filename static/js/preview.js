const video = document.getElementById("video");
const audio = document.getElementById("audio");
const playPauseButton = document.getElementById("playPauseButton");
const volumeSlider = document.getElementById("volumeSlider");
const progressBar = document.getElementById("progressBar");
const currentTimeDisplay = document.getElementById("currentTime");
const endTimeDisplay = document.getElementById("endTime");

let isPlaying = false;

function syncAudioVideo() {
  if (Math.abs(video.currentTime - audio.currentTime) > 0.1) {
    audio.currentTime = video.currentTime;
  }
}

playPauseButton.addEventListener("click", () => {
  if (isPlaying) {
    video.pause();
    audio.pause();
    playPauseButton.textContent = "Play";
  } else {
    video.play();
    audio.play();
    playPauseButton.textContent = "Pause";
  }
  isPlaying = !isPlaying;
});

volumeSlider.addEventListener("input", () => {
  const volume = volumeSlider.value;
  video.volume = volume / 100;
  audio.volume = (100 - volume) / 100;
});

function updateProgressBar() {
  const currentTime = video.currentTime;
  const duration = video.duration;
  progressBar.max = duration;
  progressBar.value = currentTime;
  currentTimeDisplay.textContent = formatTime(currentTime);
  endTimeDisplay.textContent = formatTime(duration);
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secondsRemaining = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, "0")}:${secondsRemaining
    .toString()
    .padStart(2, "0")}`;
}

progressBar.addEventListener("input", () => {
  video.currentTime = progressBar.value;
  audio.currentTime = progressBar.value;
});

video.addEventListener("timeupdate", () => {
  updateProgressBar();
  syncAudioVideo();
});

audio.addEventListener("timeupdate", syncAudioVideo);

video.addEventListener("loadedmetadata", () => {
  const duration = video.duration;
  endTimeDisplay.textContent = formatTime(duration);
  progressBar.max = duration;
});

video.volume = 0.5;
audio.volume = 0.5;
progressBar.max = video.duration || 0;
progressBar.value = 0;
currentTimeDisplay.textContent = "00:00";
endTimeDisplay.textContent = formatTime(video.duration || 0);
