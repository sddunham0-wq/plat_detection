/**
 * =====================================================
 * SISTEM AKSES KENDARAAN - JAVASCRIPT CONTROLLER
 * =====================================================
 * Penjelasan SMK: File JavaScript ini seperti "remote control" website
 * Bisa update tampilan tanpa refresh halaman (AJAX)
 *
 * Cara kerja:
 * 1. Ambil data dari server setiap beberapa detik (polling)
 * 2. Update tampilan di browser secara real-time
 * 3. Kirim perintah ke server (screenshot, manual override)
 *
 * Think of it like WhatsApp:
 * - Auto-refresh messages (detection updates)
 * - Send messages (screenshot/override commands)
 * - Update status (online/offline indicators)
 */

// =====================================================
// GLOBAL VARIABLES (Variabel Global)
// =====================================================

/**
 * Penjelasan SMK: Variabel yang bisa diakses semua fungsi
 * Seperti "papan tulis" yang bisa dilihat semua orang
 */
let detectionUpdateInterval = null;  // Timer untuk update deteksi
let isSystemActive = true;           // Status sistem aktif/tidak
const UPDATE_INTERVAL = 2000;        // Update setiap 2 detik (2000 milliseconds)

// =====================================================
// SYSTEM INITIALIZATION (Inisialisasi Sistem)
// =====================================================

/**
 * Penjelasan SMK: Fungsi yang jalan pertama kali saat halaman dibuka
 * Seperti "setup awal" sebelum mulai
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('=€ Sistem Akses Kendaraan dimulai...');

    // Mulai update deteksi otomatis
    startDetectionUpdates();

    // Check sistem status
    checkInitialSystemStatus();

    // Setup visibility change handler (pause ketika tab tidak aktif)
    setupVisibilityHandler();

    // Setup auto-refresh setiap 5 menit
    setupAutoRefresh();

    console.log(' Sistem siap digunakan!');
});

// =====================================================
// DETECTION UPDATE FUNCTIONS (Fungsi Update Deteksi)
// =====================================================

/**
 * Mulai update deteksi otomatis
 * Penjelasan SMK: Seperti "WhatsApp yang auto-refresh messages"
 */
function startDetectionUpdates() {
    console.log('=á Memulai auto-update deteksi...');

    // Clear interval lama kalau ada
    if (detectionUpdateInterval) {
        clearInterval(detectionUpdateInterval);
    }

    // Set interval baru
    detectionUpdateInterval = setInterval(updateDetectionStatus, UPDATE_INTERVAL);

    // Jalankan sekali langsung
    updateDetectionStatus();
}

/**
 * Stop update deteksi
 * Penjelasan SMK: Matikan auto-update (untuk hemat resource)
 */
function stopDetectionUpdates() {
    if (detectionUpdateInterval) {
        clearInterval(detectionUpdateInterval);
        detectionUpdateInterval = null;
        console.log('ø Auto-update dihentikan');
    }
}

/**
 * Update status deteksi dari server
 * Penjelasan SMK: "Tanya" ke server apakah ada plat terdeteksi
 */
function updateDetectionStatus() {
    // Kalau sistem tidak aktif, skip
    if (!isSystemActive) {
        return;
    }

    // Fetch data dari API (seperti "telepon" ke server)
    fetch('/api/latest_detection')
        .then(response => {
            // Convert response ke JSON
            if (!response.ok) {
                throw new Error('Server error: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // Update tampilan dengan data yang didapat
            updateDetectionPanel(data);
            updateGateStatus(data);
            updateSystemIndicators(data);
        })
        .catch(error => {
            console.error('L Error fetching detection:', error);
            handleConnectionError();
        });
}

/**
 * Update panel deteksi dengan hasil terbaru
 * Penjelasan SMK: Ubah tampilan di "Detection Panel"
 */
function updateDetectionPanel(data) {
    const statusDiv = document.getElementById('detectionStatus');

    if (!statusDiv) {
        console.warn('  Detection panel tidak ditemukan');
        return;
    }

    if (data.status === 'no_detection') {
        // Tidak ada deteksi
        statusDiv.innerHTML = `
            <div class="text-center">
                <i class="fas fa-eye fa-2x mb-3"></i>
                <p>Menunggu deteksi plat nomor...</p>
                <small>Sistem akan otomatis mendeteksi setiap ${UPDATE_INTERVAL / 1000} detik</small>
            </div>
        `;
    } else {
        // Ada deteksi plat
        const confidence = Math.round(data.confidence * 100);

        // Tentukan warna berdasarkan confidence
        let confidenceColor = 'danger';
        if (confidence > 80) {
            confidenceColor = 'success';
        } else if (confidence > 60) {
            confidenceColor = 'warning';
        }

        // Format waktu deteksi
        const detectionTime = new Date(data.timestamp).toLocaleTimeString('id-ID');

        statusDiv.innerHTML = `
            <div class="text-center">
                <i class="fas fa-check-circle fa-2x mb-3 text-success"></i>
                <h4 class="mb-3">${data.plate_text}</h4>
                <div class="progress mb-2" style="height: 25px;">
                    <div class="progress-bar bg-${confidenceColor}"
                         role="progressbar"
                         style="width: ${confidence}%"
                         aria-valuenow="${confidence}"
                         aria-valuemin="0"
                         aria-valuemax="100">
                        ${confidence}% Yakin
                    </div>
                </div>
                <small class="text-muted">
                    <i class="fas fa-clock me-1"></i>
                    Terdeteksi: ${detectionTime}
                </small>
            </div>
        `;
    }
}

/**
 * Update status palang gate
 * Penjelasan SMK: Update tampilan "Gate Terbuka/Tertutup"
 */
function updateGateStatus(data) {
    const gateDiv = document.getElementById('gateStatus');
    const gateIndicator = document.getElementById('gateStatusIndicator');

    if (!gateDiv || !gateIndicator) {
        return;
    }

    // Cek status gate dari server
    const gateStatus = data.system_status?.gate_status || 'closed';

    if (gateStatus === 'opened') {
        // Gate terbuka
        gateDiv.className = 'gate-status gate-open';
        gateDiv.innerHTML = '=â GATE TERBUKA - Akses diberikan';

        gateIndicator.className = 'badge bg-success p-2';
        gateIndicator.innerHTML = '<i class="fas fa-door-open"></i> Gate Open';
    } else {
        // Gate tertutup
        gateDiv.className = 'gate-status gate-closed';
        gateDiv.innerHTML = '=4 GATE TERTUTUP - Menunggu kendaraan...';

        gateIndicator.className = 'badge bg-secondary p-2';
        gateIndicator.innerHTML = '<i class="fas fa-door-closed"></i> Gate Closed';
    }
}

/**
 * Update indikator status sistem
 * Penjelasan SMK: Update badge "Camera OK", "Database OK", dll
 */
function updateSystemIndicators(data) {
    const cameraIndicator = document.getElementById('cameraStatusIndicator');
    const detectionIndicator = document.getElementById('detectionStatusIndicator');

    if (!cameraIndicator || !detectionIndicator) {
        return;
    }

    const systemStatus = data.system_status || {};

    // Camera status
    if (systemStatus.camera_connected) {
        cameraIndicator.className = 'badge bg-success p-2';
        cameraIndicator.innerHTML = '<i class="fas fa-camera"></i> Camera OK';
    } else {
        cameraIndicator.className = 'badge bg-danger p-2';
        cameraIndicator.innerHTML = '<i class="fas fa-camera-slash"></i> Camera Error';
    }

    // Detection status
    if (data.status === 'detected') {
        detectionIndicator.className = 'badge bg-success p-2';
        detectionIndicator.innerHTML = '<i class="fas fa-search"></i> Detecting';
    } else {
        detectionIndicator.className = 'badge bg-info p-2';
        detectionIndicator.innerHTML = '<i class="fas fa-search"></i> Standby';
    }
}

// =====================================================
// CAMERA CONTROL FUNCTIONS (Fungsi Kontrol Kamera)
// =====================================================

/**
 * Ambil screenshot dari kamera
 * Penjelasan SMK: Seperti tombol "Print Screen" di keyboard
 */
function takeScreenshot() {
    const btn = event.target.closest('button');  // Get the button element

    // Disable button sementara
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';

    console.log('=ø Mengambil screenshot...');

    // Kirim request ke server
    fetch('/api/screenshot')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showNotification(
                    ` Screenshot berhasil disimpan: ${data.filename}`,
                    'success'
                );
                console.log(' Screenshot saved:', data.path);
            } else {
                showNotification(
                    `L Error: ${data.message}`,
                    'danger'
                );
                console.error('L Screenshot error:', data.message);
            }
        })
        .catch(error => {
            showNotification(
                `L Error taking screenshot: ${error.message}`,
                'danger'
            );
            console.error('L Screenshot error:', error);
        })
        .finally(() => {
            // Enable button kembali
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-camera me-2"></i>Screenshot';
        });
}

/**
 * Manual override - buka gate secara manual
 * Penjelasan SMK: Tombol "Emergency" untuk satpam buka gate manual
 */
function manualOverride() {
    // Konfirmasi dulu
    if (!confirm('Yakin ingin membuka gate secara manual?')) {
        return;
    }

    const btn = event.target.closest('button');

    // Disable button sementara
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Opening...';

    console.log('=á Manual override activated...');

    // Kirim request ke server
    fetch('/api/manual_override')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showNotification(data.message, 'warning');
                console.log(' Manual override success');

                // Update gate status langsung
                const gateDiv = document.getElementById('gateStatus');
                if (gateDiv) {
                    gateDiv.className = 'gate-status gate-open';
                    gateDiv.innerHTML = '=á MANUAL OVERRIDE - Gate dibuka manual';
                }
            } else {
                showNotification(`L Error: ${data.message}`, 'danger');
                console.error('L Manual override error:', data.message);
            }
        })
        .catch(error => {
            showNotification(`L Error: ${error.message}`, 'danger');
            console.error('L Manual override error:', error);
        })
        .finally(() => {
            // Enable button kembali
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-unlock me-2"></i>Manual Override';
        });
}

/**
 * Refresh kamera
 * Penjelasan SMK: Reload video feed dari kamera
 */
function refreshCamera() {
    console.log('= Refreshing camera...');

    const videoFeed = document.getElementById('videoFeed');

    if (videoFeed) {
        // Add timestamp untuk bypass cache
        const timestamp = new Date().getTime();
        videoFeed.src = '/video_feed?' + timestamp;

        showNotification('=ù Camera refreshed', 'info');
    }
}

/**
 * Handle error kamera
 * Penjelasan SMK: Tampilkan pesan error kalau kamera tidak tersedia
 */
function handleCameraError(img) {
    console.error('L Camera error detected');

    // Set placeholder image
    img.src = 'data:image/svg+xml,' + encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
            <rect width="800" height="450" fill="#f8f9fa"/>
            <text x="400" y="225" text-anchor="middle" fill="#666" font-size="24" font-family="Arial">
                =ù Camera tidak tersedia
            </text>
            <text x="400" y="260" text-anchor="middle" fill="#999" font-size="16" font-family="Arial">
                Silakan periksa koneksi kamera
            </text>
        </svg>
    `);

    // Update status indicator
    const cameraStatus = document.getElementById('cameraStatus');
    if (cameraStatus) {
        cameraStatus.innerHTML = '<i class="fas fa-circle text-danger"></i> OFFLINE';
    }
}

// =====================================================
// SYSTEM STATUS FUNCTIONS (Fungsi Status Sistem)
// =====================================================

/**
 * Check sistem status awal
 * Penjelasan SMK: Cek kondisi sistem saat pertama kali buka halaman
 */
function checkInitialSystemStatus() {
    console.log('= Checking system status...');

    fetch('/api/system_status')
        .then(response => response.json())
        .then(data => {
            console.log(' System status:', data);
            updateSystemIndicators({ system_status: data });
        })
        .catch(error => {
            console.error('L Error checking system status:', error);
        });
}

/**
 * Handle connection error
 * Penjelasan SMK: Tampilkan error kalau koneksi ke server putus
 */
function handleConnectionError() {
    const cameraIndicator = document.getElementById('cameraStatusIndicator');

    if (cameraIndicator) {
        cameraIndicator.className = 'badge bg-danger p-2';
        cameraIndicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Connection Error';
    }

    showNotification('  Koneksi ke server terputus', 'warning');
}

// =====================================================
// NOTIFICATION FUNCTIONS (Fungsi Notifikasi)
// =====================================================

/**
 * Tampilkan notifikasi toast
 * Penjelasan SMK: Munculkan "popup kecil" di pojok kanan atas
 */
function showNotification(message, type = 'info') {
    // Buat atau dapatkan container toast
    let toastContainer = document.querySelector('.toast-container');

    if (!toastContainer) {
        toastContainer = createToastContainer();
    }

    // Buat elemen toast
    const toast = document.createElement('div');
    toast.className = 'toast show';
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    // Icon berdasarkan type
    const iconMap = {
        'success': 'check-circle',
        'danger': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };

    const icon = iconMap[type] || 'info-circle';

    toast.innerHTML = `
        <div class="toast-header">
            <i class="fas fa-${icon} text-${type} me-2"></i>
            <strong class="me-auto">Sistem</strong>
            <small class="text-muted">Baru saja</small>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;

    // Append ke container
    toastContainer.appendChild(toast);

    // Auto remove setelah 5 detik
    setTimeout(() => {
        toast.classList.add('hiding');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 5000);
}

/**
 * Buat container untuk toast
 * Penjelasan SMK: Buat "tempat" untuk taruh notifikasi
 */
function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '1050';
    document.body.appendChild(container);
    return container;
}

// =====================================================
// PAGE VISIBILITY & AUTO-REFRESH (Visibility & Refresh Otomatis)
// =====================================================

/**
 * Setup visibility change handler
 * Penjelasan SMK: Pause update kalau user pindah tab (hemat resource)
 */
function setupVisibilityHandler() {
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'visible') {
            // Tab aktif kembali
            console.log('=A Tab active - resuming updates...');

            if (!detectionUpdateInterval) {
                startDetectionUpdates();
            }
            isSystemActive = true;
        } else {
            // Tab tidak aktif
            console.log('=4 Tab hidden - pausing updates...');
            isSystemActive = false;
        }
    });
}

/**
 * Setup auto-refresh halaman
 * Penjelasan SMK: Refresh halaman otomatis setiap 5 menit (untuk data fresh)
 */
function setupAutoRefresh() {
    const AUTO_REFRESH_TIME = 5 * 60 * 1000;  // 5 menit

    setInterval(() => {
        // Hanya refresh kalau tab aktif
        if (document.visibilityState === 'visible') {
            console.log('= Auto-refreshing page...');
            location.reload();
        }
    }, AUTO_REFRESH_TIME);
}

// =====================================================
// UTILITY FUNCTIONS (Fungsi Bantuan)
// =====================================================

/**
 * Format timestamp ke waktu lokal
 * Penjelasan SMK: Ubah format waktu jadi mudah dibaca
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('id-ID', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * Debounce function untuk performance
 * Penjelasan SMK: "Rem" untuk fungsi yang dipanggil terlalu sering
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// =====================================================
// ERROR HANDLING (Penanganan Error)
// =====================================================

/**
 * Global error handler
 * Penjelasan SMK: Tangkap semua error yang tidak terduga
 */
window.addEventListener('error', function(event) {
    console.error('=¨ Global error:', event.error);

    // Tampilkan notifikasi ke user (optional)
    // showNotification('Terjadi kesalahan sistem', 'danger');
});

/**
 * Unhandled promise rejection handler
 * Penjelasan SMK: Tangkap error dari Promise yang tidak di-catch
 */
window.addEventListener('unhandledrejection', function(event) {
    console.error('=¨ Unhandled promise rejection:', event.reason);
});

// =====================================================
// CLEANUP ON PAGE UNLOAD (Cleanup saat tutup halaman)
// =====================================================

/**
 * Cleanup saat halaman ditutup
 * Penjelasan SMK: Bersihkan resource sebelum tutup
 */
window.addEventListener('beforeunload', function() {
    stopDetectionUpdates();
    console.log('=K Cleaning up resources...');
});

// =====================================================
// CONSOLE LOGGING (Logging untuk Development)
// =====================================================

console.log(`
TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW
Q  =— SISTEM AKSES KENDARAAN SMK                            Q
Q  =Å Version 1.0.0 - 2024                                  Q
Q  =» JavaScript Controller Loaded                          Q
Q                                                            Q
Q  Fitur:                                                    Q
Q   Real-time detection updates                           Q
Q   Camera screenshot                                     Q
Q   Manual gate override                                  Q
Q   Auto-refresh & visibility handling                    Q
Q   Toast notifications                                   Q
ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
`);

// =====================================================
// END OF JAVASCRIPT CONTROLLER
// =====================================================
