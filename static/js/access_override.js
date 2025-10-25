/**
 * Access Override Control Panel
 * Handles manual override decisions with anti-spam alerts
 */

// WebSocket connection
let socket = null;

// Alert management
const alertContainer = document.getElementById('alertContainer');
const maxVisibleAlerts = 3;
let activeAlerts = [];

// Audio management
let audioEnabled = true;
let audioVolume = 0.8;
const audioCache = {};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initWebSocket();
    initEventListeners();
    loadAlertSettings();
    startStatsRefresh();
    preloadAudioFiles();
});

/**
 * Initialize WebSocket connection
 */
function initWebSocket() {
    socket = io();

    socket.on('connect', function() {
        console.log('✅ WebSocket connected');
        showToast('Connected to server', 'success');
    });

    socket.on('disconnect', function() {
        console.log('❌ WebSocket disconnected');
        showToast('Disconnected from server', 'warning');
    });

    // Override alert events
    socket.on('override_alert', function(data) {
        console.log('🔔 Override alert:', data);
        handleOverrideAlert(data);
    });

    // Pending review notifications
    socket.on('pending_manual_review', function(data) {
        console.log('⚠️ Pending review:', data);
        addPendingReview(data);
    });

    // Access control results
    socket.on('access_control_result', function(data) {
        console.log('🚦 Access result:', data);
        handleAccessResult(data);
    });
}

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Edit plate buttons
    document.querySelectorAll('.btn-edit-plate').forEach(btn => {
        btn.addEventListener('click', handleEditPlate);
    });

    // Approve buttons
    document.querySelectorAll('.btn-approve').forEach(btn => {
        btn.addEventListener('click', handleApprove);
    });

    // Reject buttons
    document.querySelectorAll('.btn-reject').forEach(btn => {
        btn.addEventListener('click', handleReject);
    });

    // Save correction button
    document.getElementById('saveCorrection')?.addEventListener('click', saveCorrection);

    // Confirm override button
    document.getElementById('confirmOverride')?.addEventListener('click', confirmOverride);

    // Save settings button
    document.getElementById('saveSettings')?.addEventListener('click', saveSettings);

    // Volume slider
    document.getElementById('settingVolume')?.addEventListener('input', function(e) {
        document.getElementById('volumeDisplay').textContent = e.target.value + '%';
    });
}

/**
 * Handle edit plate button click
 */
function handleEditPlate(e) {
    const btn = e.currentTarget;
    const detectionId = btn.dataset.id;
    const plateNumber = btn.dataset.plate;

    document.getElementById('editDetectionId').value = detectionId;
    document.getElementById('editOriginalPlate').value = plateNumber;
    document.getElementById('editCorrectedPlate').value = plateNumber;

    const modal = new bootstrap.Modal(document.getElementById('editPlateModal'));
    modal.show();
}

/**
 * Save plate correction
 */
async function saveCorrection() {
    const detectionId = document.getElementById('editDetectionId').value;
    const originalPlate = document.getElementById('editOriginalPlate').value;
    const correctedPlate = document.getElementById('editCorrectedPlate').value.trim().toUpperCase();
    const reason = document.getElementById('editReason').value;

    if (!correctedPlate) {
        showToast('Please enter corrected plate number', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/override/correct-plate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                detection_id: detectionId,
                original_plate: originalPlate,
                corrected_plate: correctedPlate,
                reason: reason
            })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Plate corrected: ${originalPlate} → ${correctedPlate}`, 'success');
            bootstrap.Modal.getInstance(document.getElementById('editPlateModal')).hide();

            // Update UI
            updatePlateInUI(detectionId, correctedPlate);
        } else {
            showToast('Error: ' + data.error, 'danger');
        }
    } catch (error) {
        console.error('Error saving correction:', error);
        showToast('Error saving correction', 'danger');
    }
}

/**
 * Handle approve button click
 */
function handleApprove(e) {
    const btn = e.currentTarget;
    const detectionId = btn.dataset.id;
    const plateNumber = btn.dataset.plate;

    showOverrideModal('approved', detectionId, plateNumber);
}

/**
 * Handle reject button click
 */
function handleReject(e) {
    const btn = e.currentTarget;
    const detectionId = btn.dataset.id;
    const plateNumber = btn.dataset.plate;

    showOverrideModal('rejected', detectionId, plateNumber);
}

/**
 * Show override modal
 */
function showOverrideModal(decision, detectionId, plateNumber) {
    const modal = document.getElementById('overrideModal');
    const header = document.getElementById('overrideModalHeader');
    const title = document.getElementById('overrideModalTitle');
    const confirmBtn = document.getElementById('confirmOverride');
    const durationGroup = document.getElementById('durationGroup');

    document.getElementById('overrideDetectionId').value = detectionId;
    document.getElementById('overridePlate').value = plateNumber;
    document.getElementById('overrideDecision').value = decision;
    document.getElementById('overridePlateDisplay').textContent = plateNumber;
    document.getElementById('overridePin').value = '';

    if (decision === 'approved') {
        header.className = 'modal-header bg-success text-white';
        title.innerHTML = '<i class="bi bi-check-circle"></i> Approve Access';
        confirmBtn.className = 'btn btn-success';
        confirmBtn.innerHTML = '<i class="bi bi-check-lg"></i> Approve Access';
        durationGroup.style.display = 'block';
    } else {
        header.className = 'modal-header bg-danger text-white';
        title.innerHTML = '<i class="bi bi-x-circle"></i> Deny Access';
        confirmBtn.className = 'btn btn-danger';
        confirmBtn.innerHTML = '<i class="bi bi-x-lg"></i> Deny Access';
        durationGroup.style.display = 'none';
    }

    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * Confirm override decision
 */
async function confirmOverride() {
    const detectionId = document.getElementById('overrideDetectionId').value;
    const plateNumber = document.getElementById('overridePlate').value;
    const decision = document.getElementById('overrideDecision').value;
    const reason = document.getElementById('overrideReason').value;
    const duration = document.getElementById('overrideDuration').value;
    const pin = document.getElementById('overridePin').value;

    if (!pin) {
        showToast('Please enter PIN', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/override/access-decision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                detection_id: detectionId,
                plate_number: plateNumber,
                decision: decision,
                reason: reason,
                duration: duration,
                pin: pin
            })
        });

        const data = await response.json();

        if (data.success) {
            const action = decision === 'approved' ? 'APPROVED' : 'REJECTED';
            showToast(`Access ${action}: ${plateNumber}`, decision === 'approved' ? 'success' : 'danger');
            playSound(decision === 'approved' ? 'access_granted' : 'access_denied');

            bootstrap.Modal.getInstance(document.getElementById('overrideModal')).hide();

            // Remove from pending reviews
            removePendingReview(detectionId);

            // Add to recent decisions
            addRecentDecision(plateNumber, decision, reason);

            // Update stats
            refreshStats();
        } else {
            if (data.error === 'Invalid PIN') {
                showToast('Invalid PIN!', 'danger');
                document.getElementById('overridePin').value = '';
                document.getElementById('overridePin').focus();
            } else {
                showToast('Error: ' + data.error, 'danger');
            }
        }
    } catch (error) {
        console.error('Error confirming override:', error);
        showToast('Error processing decision', 'danger');
    }
}

/**
 * Handle override alert from WebSocket
 */
function handleOverrideAlert(data) {
    const { type, plate_number, reason, priority } = data;

    let alertType = 'info';
    let message = '';
    let soundFile = null;

    if (type === 'manual_approved') {
        alertType = 'success';
        message = `✅ ACCESS GRANTED: ${plate_number}`;
        soundFile = 'access_granted';
    } else if (type === 'manual_rejected') {
        alertType = 'danger';
        message = `❌ ACCESS DENIED: ${plate_number}`;
        soundFile = 'access_denied';
    } else if (type === 'manual_required') {
        alertType = 'warning';
        message = `⚠️ MANUAL REVIEW REQUIRED: ${plate_number}`;
        soundFile = 'manual_required';
    }

    if (reason) {
        message += `<br><small>${reason}</small>`;
    }

    showAlert(message, alertType, priority);

    if (soundFile) {
        playSound(soundFile);
    }
}

/**
 * Handle access control result
 */
function handleAccessResult(data) {
    const { access, plate_number, owner_name } = data;

    let message = '';
    let alertType = 'info';
    let soundFile = null;

    if (access === 'granted') {
        message = `✅ ${plate_number}`;
        if (owner_name) message += ` - ${owner_name}`;
        alertType = 'success';
        soundFile = 'access_granted';
    } else {
        message = `❌ ${plate_number} - ACCESS DENIED`;
        alertType = 'danger';
        soundFile = 'access_denied';
    }

    showAlert(message, alertType, 'MEDIUM');

    if (soundFile) {
        playSound(soundFile);
    }
}

/**
 * Show alert notification
 */
function showAlert(message, type = 'info', priority = 'MEDIUM') {
    // Check if should show based on settings
    // (implement priority filtering here based on user settings)

    // Remove excess alerts
    while (activeAlerts.length >= maxVisibleAlerts) {
        const oldestAlert = activeAlerts.shift();
        oldestAlert.remove();
    }

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-notification`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    alertContainer.appendChild(alertDiv);
    activeAlerts.push(alertDiv);

    // Auto-dismiss after configured time
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.classList.remove('show');
            setTimeout(() => {
                alertDiv.remove();
                activeAlerts = activeAlerts.filter(a => a !== alertDiv);
            }, 300);
        }
    }, 5000); // Default 5 seconds
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Simple toast using Bootstrap alerts
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
    toast.style.zIndex = '9999';
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}

/**
 * Preload audio files
 */
function preloadAudioFiles() {
    const audioFiles = ['access_granted', 'access_denied', 'manual_required', 'manual_override'];

    audioFiles.forEach(file => {
        const audio = new Audio(`/static/sounds/${file}.mp3`);
        audio.volume = audioVolume;
        audioCache[file] = audio;
    });
}

/**
 * Play sound
 */
function playSound(soundName) {
    if (!audioEnabled) return;

    const audio = audioCache[soundName];
    if (audio) {
        audio.currentTime = 0;
        audio.volume = audioVolume;
        audio.play().catch(e => console.log('Audio play failed:', e));
    }
}

/**
 * Update plate number in UI
 */
function updatePlateInUI(detectionId, newPlate) {
    const card = document.querySelector(`[data-review-id="${detectionId}"]`);
    if (card) {
        const plateElement = card.querySelector('h5');
        if (plateElement) {
            plateElement.textContent = newPlate;
        }
    }
}

/**
 * Remove pending review from UI
 */
function removePendingReview(detectionId) {
    const card = document.querySelector(`[data-review-id="${detectionId}"]`);
    if (card) {
        card.style.transition = 'all 0.3s';
        card.style.opacity = '0';
        card.style.transform = 'translateX(-100%)';

        setTimeout(() => {
            card.remove();

            // Update count
            const pendingCount = document.querySelectorAll('.pending-review-card').length;
            document.getElementById('pendingCount').textContent = pendingCount;

            // Show empty state if no reviews
            if (pendingCount === 0) {
                const container = document.getElementById('pendingReviewsContainer');
                container.innerHTML = `
                    <div class="empty-state">
                        <i class="bi bi-check-circle"></i>
                        <h4>No Pending Reviews</h4>
                        <p class="text-muted">All detections have been processed</p>
                    </div>
                `;
            }
        }, 300);
    }
}

/**
 * Add pending review to UI
 */
function addPendingReview(data) {
    // Implement adding new pending review card dynamically
    // (similar to the template structure)
    console.log('Adding pending review:', data);

    // Reload pending reviews
    refreshPendingReviews();
}

/**
 * Add recent decision to panel
 */
function addRecentDecision(plateNumber, decision, reason) {
    const container = document.getElementById('recentDecisionsContainer');

    // Remove empty state if exists
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    const decisionDiv = document.createElement('div');
    decisionDiv.className = 'decision-item';
    decisionDiv.innerHTML = `
        <i class="bi ${decision === 'approved' ? 'bi-check-circle-fill decision-icon approved' : 'bi-x-circle-fill decision-icon rejected'}"></i>
        <div class="flex-grow-1">
            <strong>${plateNumber}</strong>
            <br><small class="text-muted">${reason} - ${new Date().toLocaleTimeString()}</small>
        </div>
    `;

    container.insertBefore(decisionDiv, container.firstChild);

    // Keep only last 10 decisions
    while (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
}

/**
 * Load alert settings
 */
async function loadAlertSettings() {
    try {
        const response = await fetch('/api/override/settings?user_id=default');
        const data = await response.json();

        if (data.success) {
            const settings = data.settings;
            audioEnabled = settings.enable_audio || true;
            audioVolume = settings.audio_volume || 0.8;

            console.log('Alert settings loaded:', settings);
        }
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

/**
 * Save settings
 */
async function saveSettings() {
    const settings = {
        enable_audio: document.getElementById('settingEnableAudio').checked,
        audio_volume: parseInt(document.getElementById('settingVolume').value) / 100,
        sound_denied: document.getElementById('settingSoundDenied').checked,
        sound_granted_auto: document.getElementById('settingSoundGrantedAuto').checked,
        sound_granted_manual: document.getElementById('settingSoundGrantedManual').checked,
        sound_manual_required: document.getElementById('settingSoundManualRequired').checked,
        auto_dismiss_seconds: parseInt(document.getElementById('settingAutoDismiss').value),
        max_visible_alerts: parseInt(document.getElementById('settingMaxAlerts').value),
        enable_grouping: document.getElementById('settingEnableGrouping').checked,
        show_critical: document.getElementById('settingShowCritical').checked,
        show_high: document.getElementById('settingShowHigh').checked,
        show_medium: document.getElementById('settingShowMedium').checked,
        show_low: document.getElementById('settingShowLow').checked,
        enable_quiet_hours: document.getElementById('settingEnableQuietHours').checked,
        quiet_start_time: document.getElementById('settingQuietStart').value,
        quiet_end_time: document.getElementById('settingQuietEnd').value
    };

    try {
        const response = await fetch('/api/override/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_id: 'default',
                settings: settings
            })
        });

        const data = await response.json();

        if (data.success) {
            showToast('Settings saved successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('settingsModal')).hide();

            // Update local settings
            audioEnabled = settings.enable_audio;
            audioVolume = settings.audio_volume;
        } else {
            showToast('Error saving settings: ' + data.error, 'danger');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        showToast('Error saving settings', 'danger');
    }
}

/**
 * Refresh statistics
 */
async function refreshStats() {
    try {
        const response = await fetch('/api/override/stats');
        const data = await response.json();

        if (data.success) {
            const stats = data.stats;
            document.getElementById('statTotalDetections').textContent = stats.total_detections || 0;
            document.getElementById('statGranted').textContent = stats.granted_count || 0;
            document.getElementById('statDenied').textContent = stats.denied_count || 0;
            document.getElementById('statOverrides').textContent = stats.manual_override_count || 0;
            document.getElementById('detectionRate').textContent = stats.detections_per_hour || 0;

            const busyBadge = document.getElementById('statBusyMode');
            if (stats.is_busy) {
                busyBadge.innerHTML = '<span class="badge bg-warning">Active</span>';
            } else {
                busyBadge.innerHTML = '<span class="badge bg-secondary">Inactive</span>';
            }
        }
    } catch (error) {
        console.error('Error refreshing stats:', error);
    }
}

/**
 * Refresh pending reviews
 */
async function refreshPendingReviews() {
    try {
        const response = await fetch('/api/override/pending-reviews?limit=20');
        const data = await response.json();

        if (data.success) {
            // Update count
            document.getElementById('pendingCount').textContent = data.count;

            // TODO: Update pending reviews container with new data
        }
    } catch (error) {
        console.error('Error refreshing pending reviews:', error);
    }
}

/**
 * Start auto-refresh of stats
 */
function startStatsRefresh() {
    // Refresh stats every 5 seconds
    setInterval(refreshStats, 5000);

    // Refresh pending reviews every 10 seconds
    setInterval(refreshPendingReviews, 10000);
}
