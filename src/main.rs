use anyhow::{Context, Result};
use arboard::Clipboard;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use mistralrs::{AudioInput, IsqType, MultimodalMessages, MultimodalModelBuilder, TextMessageRole};
use std::io::Cursor;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

const TARGET_RATE: u32 = 16_000;
const VAD_THRESHOLD: f32 = 0.015;
const SILENCE_CHUNKS_TO_STOP: usize = 30;
const MIN_SPEECH_SECONDS: f32 = 0.3;
const MAX_SPEECH_SECONDS: f32 = 28.0;
const MIN_PADDED_SECONDS: u32 = 4;

// ---------------------------------------------------------------------------
// Global hotkey state
// ---------------------------------------------------------------------------

static GLOBAL_PTT_ACTIVE: AtomicBool = AtomicBool::new(false);
static GLOBAL_PTT_RELEASED: AtomicBool = AtomicBool::new(false);
static HOTKEY_VK: AtomicU32 = AtomicU32::new(0x78); // F9

fn hotkey_options() -> Vec<(&'static str, u32)> {
    vec![
        ("F5", 0x74),
        ("F6", 0x75),
        ("F7", 0x76),
        ("F8", 0x77),
        ("F9", 0x78),
        ("F10", 0x79),
        ("F11", 0x7A),
        ("F12", 0x7B),
        ("Scroll Lock", 0x91),
        ("Pause", 0x13),
    ]
}

fn hotkey_name(vk: u32) -> &'static str {
    hotkey_options()
        .iter()
        .find(|(_, code)| *code == vk)
        .map(|(name, _)| *name)
        .unwrap_or("F9")
}

fn start_global_hotkey_thread() {
    std::thread::spawn(|| {
        let mut was_pressed = false;
        loop {
            let vk = HOTKEY_VK.load(Ordering::SeqCst) as i32;
            let pressed = unsafe { GetAsyncKeyState(vk) } & (0x8000u16 as i16) != 0;

            if pressed && !was_pressed {
                GLOBAL_PTT_ACTIVE.store(true, Ordering::SeqCst);
            } else if !pressed && was_pressed {
                GLOBAL_PTT_ACTIVE.store(false, Ordering::SeqCst);
                GLOBAL_PTT_RELEASED.store(true, Ordering::SeqCst);
            }

            was_pressed = pressed;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    });
}

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }
    let ratio = source_rate as f64 / target_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac as f32
        })
        .collect()
}

fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::new(&mut buf, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(buf.into_inner())
}

fn is_valid_transcription(text: &str) -> bool {
    !text.is_empty() && !text.contains('<') && !text.contains('>')
}

fn paste_text(text: &str) {
    if let Ok(mut clip) = Clipboard::new() {
        if clip.set_text(text).is_ok() {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if let Ok(mut enigo) = Enigo::new(&Settings::default()) {
                let _ = enigo.key(Key::Control, Direction::Press);
                let _ = enigo.key(Key::Unicode('v'), Direction::Click);
                let _ = enigo.key(Key::Control, Direction::Release);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
enum InputMode {
    PushToTalk,
    Vad,
}

#[derive(Clone, PartialEq)]
enum AppStatus {
    LoadingModel,
    Ready,
    Recording,
    Transcribing,
    Error(String),
}

impl std::fmt::Display for AppStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppStatus::LoadingModel => write!(f, "Loading model..."),
            AppStatus::Ready => write!(f, "Ready"),
            AppStatus::Recording => write!(f, "Recording..."),
            AppStatus::Transcribing => write!(f, "Transcribing..."),
            AppStatus::Error(e) => write!(f, "Error: {e}"),
        }
    }
}

struct SharedState {
    status: AppStatus,
    transcriptions: Vec<String>,
    current_rms: f32,
    input_mode: InputMode,
    model_ready: bool,
    auto_paste: bool,
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------

struct TranscriberApp {
    state: Arc<Mutex<SharedState>>,
    selected_hotkey_idx: usize,
}

impl eframe::App for TranscriberApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        let (status, rms, transcriptions, input_mode, model_ready, auto_paste) = {
            let s = self.state.lock().unwrap();
            (
                s.status.clone(),
                s.current_rms,
                s.transcriptions.clone(),
                s.input_mode.clone(),
                s.model_ready,
                s.auto_paste,
            )
        };

        // Status bar
        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let (color, label) = match &status {
                    AppStatus::LoadingModel => (egui::Color32::YELLOW, "LOADING"),
                    AppStatus::Ready => (egui::Color32::GREEN, "READY"),
                    AppStatus::Recording => (egui::Color32::RED, "REC"),
                    AppStatus::Transcribing => (egui::Color32::LIGHT_BLUE, "TRANSCRIBING"),
                    AppStatus::Error(_) => (egui::Color32::RED, "ERROR"),
                };

                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, color);
                ui.label(egui::RichText::new(label).color(color).strong().size(13.0));

                ui.separator();

                let mut is_ptt = input_mode == InputMode::PushToTalk;
                if ui.selectable_label(is_ptt, "Push-to-Talk").clicked() {
                    is_ptt = true;
                }
                if ui.selectable_label(!is_ptt, "VAD (auto)").clicked() {
                    is_ptt = false;
                }
                {
                    let mut s = self.state.lock().unwrap();
                    s.input_mode = if is_ptt {
                        InputMode::PushToTalk
                    } else {
                        InputMode::Vad
                    };
                }

                ui.separator();

                let mut paste_on = auto_paste;
                if ui.checkbox(&mut paste_on, "Auto-paste").changed() {
                    self.state.lock().unwrap().auto_paste = paste_on;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let meter_w = 100.0;
                    let meter_h = 12.0;
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(meter_w, meter_h), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 3.0, egui::Color32::from_gray(40));

                    let level = (rms * 20.0).min(1.0);
                    let bar_color = if level > 0.6 {
                        egui::Color32::RED
                    } else if level > 0.3 {
                        egui::Color32::YELLOW
                    } else {
                        egui::Color32::GREEN
                    };
                    let bar_rect = egui::Rect::from_min_size(
                        rect.min,
                        egui::vec2(meter_w * level, meter_h),
                    );
                    ui.painter().rect_filled(bar_rect, 3.0, bar_color);
                    ui.label("Mic:");
                });
            });
            ui.add_space(2.0);
        });

        // Bottom panel: hotkey selector
        if input_mode == InputMode::PushToTalk && model_ready {
            egui::TopBottomPanel::bottom("ptt_panel").show(ctx, |ui| {
                ui.add_space(4.0);
                let is_recording = status == AppStatus::Recording;

                ui.horizontal(|ui| {
                    ui.label("Hotkey:");
                    let options = hotkey_options();
                    let current_name = hotkey_name(HOTKEY_VK.load(Ordering::SeqCst));

                    egui::ComboBox::from_id_salt("hotkey_select")
                        .selected_text(current_name)
                        .show_ui(ui, |ui| {
                            for (i, (name, vk)) in options.iter().enumerate() {
                                if ui
                                    .selectable_value(&mut self.selected_hotkey_idx, i, *name)
                                    .clicked()
                                {
                                    HOTKEY_VK.store(*vk, Ordering::SeqCst);
                                }
                            }
                        });

                    ui.separator();

                    if is_recording {
                        ui.colored_label(
                            egui::Color32::RED,
                            egui::RichText::new(">> RECORDING -- release key to transcribe <<")
                                .size(15.0)
                                .strong(),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!(
                                "Hold {} to record (works globally, any app)",
                                current_name
                            ))
                            .size(14.0)
                            .color(egui::Color32::GRAY),
                        );
                    }
                });
                ui.add_space(4.0);
            });
        }

        // Transcriptions
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Transcriptions");
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if transcriptions.is_empty() {
                        let hint = if input_mode == InputMode::PushToTalk {
                            "Hold the hotkey to speak (works in any app)..."
                        } else {
                            "Speak into your microphone..."
                        };
                        ui.label(
                            egui::RichText::new(hint)
                                .italics()
                                .color(egui::Color32::GRAY),
                        );
                    } else {
                        for (i, text) in transcriptions.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("[{}]", i + 1))
                                        .color(egui::Color32::DARK_GRAY)
                                        .small(),
                                );
                                ui.label(egui::RichText::new(text).size(15.0));
                            });
                            ui.add_space(4.0);
                        }
                    }
                });
        });
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

fn run_backend(state: Arc<Mutex<SharedState>>) {
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    rt.block_on(async move {
        let cache_dir = std::env::current_dir()
            .unwrap_or_default()
            .join("models");
        std::fs::create_dir_all(&cache_dir).ok();
        std::env::set_var("HF_HOME", &cache_dir);

        let model = match MultimodalModelBuilder::new("google/gemma-4-E4B-it")
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await
        {
            Ok(m) => m,
            Err(e) => {
                state.lock().unwrap().status =
                    AppStatus::Error(format!("Model load failed: {e}"));
                return;
            }
        };

        {
            let mut s = state.lock().unwrap();
            s.status = AppStatus::Ready;
            s.model_ready = true;
        }

        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => {
                state.lock().unwrap().status =
                    AppStatus::Error("No input device found".into());
                return;
            }
        };

        let default_config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => {
                state.lock().unwrap().status =
                    AppStatus::Error(format!("Audio config error: {e}"));
                return;
            }
        };

        let device_rate = default_config.sample_rate().0;
        let device_channels = default_config.channels();
        let min_samples =
            (MIN_SPEECH_SECONDS * device_rate as f32) as usize * device_channels as usize;
        let max_samples =
            (MAX_SPEECH_SECONDS * device_rate as f32) as usize * device_channels as usize;

        let stream_config = cpal::StreamConfig {
            channels: device_channels,
            sample_rate: cpal::SampleRate(device_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();

        let audio_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let vad_speaking: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let silence_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));

        let buf_c = audio_buf.clone();
        let vad_c = vad_speaking.clone();
        let sil_c = silence_count.clone();
        let state_c = state.clone();

        let stream = match device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let rms = compute_rms(data);

                if let Ok(mut s) = state_c.try_lock() {
                    s.current_rms = rms;
                }

                let mode = state_c
                    .try_lock()
                    .map(|s| s.input_mode.clone())
                    .unwrap_or(InputMode::PushToTalk);

                match mode {
                    InputMode::PushToTalk => {
                        let recording = GLOBAL_PTT_ACTIVE.load(Ordering::SeqCst);
                        let just_released = GLOBAL_PTT_RELEASED.swap(false, Ordering::SeqCst);
                        let mut buf = buf_c.lock().unwrap();

                        if recording {
                            buf.extend_from_slice(data);
                            if let Ok(mut s) = state_c.try_lock() {
                                if s.status != AppStatus::Transcribing {
                                    s.status = AppStatus::Recording;
                                }
                            }
                            if buf.len() >= max_samples {
                                if buf.len() >= min_samples {
                                    let _ = tx.send(buf.clone());
                                }
                                buf.clear();
                                GLOBAL_PTT_ACTIVE.store(false, Ordering::SeqCst);
                            }
                        } else if just_released {
                            if buf.len() >= min_samples {
                                let _ = tx.send(buf.clone());
                            }
                            buf.clear();
                        }
                    }
                    InputMode::Vad => {
                        let speaking = vad_c.load(Ordering::SeqCst);
                        let mut buf = buf_c.lock().unwrap();
                        let mut silence = sil_c.lock().unwrap();

                        if !speaking {
                            if rms > VAD_THRESHOLD {
                                vad_c.store(true, Ordering::SeqCst);
                                *silence = 0;
                                buf.clear();
                                buf.extend_from_slice(data);
                            }
                        } else {
                            buf.extend_from_slice(data);

                            if buf.len() >= max_samples {
                                if buf.len() >= min_samples {
                                    let _ = tx.send(buf.clone());
                                }
                                buf.clear();
                                vad_c.store(false, Ordering::SeqCst);
                                *silence = 0;
                                return;
                            }

                            if rms < VAD_THRESHOLD {
                                *silence += 1;
                                if *silence >= SILENCE_CHUNKS_TO_STOP {
                                    if buf.len() >= min_samples {
                                        let _ = tx.send(buf.clone());
                                    }
                                    buf.clear();
                                    vad_c.store(false, Ordering::SeqCst);
                                    *silence = 0;
                                }
                            } else {
                                *silence = 0;
                            }
                        }
                    }
                }
            },
            |err| eprintln!("Audio stream error: {err}"),
            None,
        ) {
            Ok(s) => s,
            Err(e) => {
                state.lock().unwrap().status =
                    AppStatus::Error(format!("Stream error: {e}"));
                return;
            }
        };

        if let Err(e) = stream.play() {
            state.lock().unwrap().status =
                AppStatus::Error(format!("Stream play error: {e}"));
            return;
        }

        // Transcription loop
        while let Some(raw_samples) = rx.recv().await {
            {
                state.lock().unwrap().status = AppStatus::Transcribing;
            }

            let mono = to_mono(&raw_samples, device_channels);
            let mut resampled = resample(&mono, device_rate, TARGET_RATE);
            let duration = resampled.len() as f32 / TARGET_RATE as f32;

            // Pad short clips with silence so the model has enough context
            let min_padded = (TARGET_RATE * MIN_PADDED_SECONDS) as usize;
            if resampled.len() < min_padded {
                resampled.resize(min_padded, 0.0);
            }

            match encode_wav(&resampled)
                .and_then(|wav| AudioInput::from_bytes(&wav).context("AudioInput creation failed"))
            {
                Ok(audio) => {
                    let messages = MultimodalMessages::new().add_multimodal_message(
                        TextMessageRole::User,
                        "The audio is in German. Transcribe the spoken words exactly in German. \
                         The audio may contain short phrases or single words. \
                         Output only the transcription, nothing else.",
                        vec![],
                        vec![audio],
                        vec![],
                    );

                    match model.send_chat_request(messages).await {
                        Ok(response) => {
                            let text = response.choices[0]
                                .message
                                .content
                                .clone()
                                .unwrap_or_default();

                            if is_valid_transcription(&text) {
                                let do_paste;
                                {
                                    let mut s = state.lock().unwrap();
                                    s.transcriptions
                                        .push(format!("[{duration:.1}s] {text}"));
                                    do_paste = s.auto_paste;
                                }
                                if do_paste {
                                    paste_text(&text);
                                }
                            } else {
                                state.lock().unwrap().transcriptions
                                    .push(format!("[{duration:.1}s] (unclear, try again)"));
                            }
                        }
                        Err(e) => {
                            state.lock().unwrap().transcriptions
                                .push(format!("[error] {e}"));
                        }
                    }
                }
                Err(e) => {
                    state.lock().unwrap().transcriptions
                        .push(format!("[error] {e}"));
                }
            }

            state.lock().unwrap().status = AppStatus::Ready;
        }
    });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> eframe::Result<()> {
    start_global_hotkey_thread();

    let shared = Arc::new(Mutex::new(SharedState {
        status: AppStatus::LoadingModel,
        transcriptions: Vec::new(),
        current_rms: 0.0,
        input_mode: InputMode::PushToTalk,
        model_ready: false,
        auto_paste: false,
    }));

    let default_idx = hotkey_options()
        .iter()
        .position(|(_, vk)| *vk == 0x78)
        .unwrap_or(4);

    let backend_state = shared.clone();
    std::thread::spawn(move || run_backend(backend_state));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([650.0, 420.0])
            .with_title("Gemma 4 Audio Transcriber"),
        ..Default::default()
    };

    eframe::run_native(
        "Gemma 4 Audio Transcriber",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(TranscriberApp {
                state: shared,
                selected_hotkey_idx: default_idx,
            }))
        }),
    )
}
