"""
Alert Anti-Spam System
Prevent notification spam dengan intelligent filtering dan grouping
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = 1    # Manual override required - cannot be suppressed
    HIGH = 2        # Access denied - show with sound
    MEDIUM = 3      # Access granted (registered) - show notification only
    LOW = 4         # Auto-processed - log only


@dataclass
class AlertEvent:
    """Alert event data structure"""
    plate_number: str
    decision: str  # 'granted', 'denied', 'pending'
    priority: AlertPriority
    confidence: float
    timestamp: float = field(default_factory=time.time)
    reason: Optional[str] = None
    manual_override: bool = False
    owner_name: Optional[str] = None
    vehicle_type: Optional[str] = None


class DuplicateDetectionFilter:
    """
    Prevent duplicate alerts untuk plat yang sama dalam cooldown period
    """
    def __init__(self, cooldown_period: int = 30):
        """
        Args:
            cooldown_period: Seconds to wait before allowing same plate alert
        """
        self.cooldown_period = cooldown_period
        self.recent_detections: Dict[str, float] = {}  # {plate_number: last_alert_time}
        self._lock = threading.Lock()

    def should_alert(self, plate_number: str) -> bool:
        """
        Check if alert should be shown untuk plate number ini

        Args:
            plate_number: License plate number

        Returns:
            True if alert should be shown, False if dalam cooldown period
        """
        with self._lock:
            now = time.time()
            last_alert = self.recent_detections.get(plate_number, 0)

            # Jika sudah lewat cooldown period, allow alert
            if now - last_alert >= self.cooldown_period:
                self.recent_detections[plate_number] = now
                return True

            return False

    def force_alert(self, plate_number: str):
        """Force update last alert time (for manual overrides)"""
        with self._lock:
            self.recent_detections[plate_number] = time.time()

    def clear_plate(self, plate_number: str):
        """Remove plate dari tracking (useful for testing)"""
        with self._lock:
            self.recent_detections.pop(plate_number, None)

    def cleanup_old_entries(self, max_age: int = 3600):
        """Remove entries older than max_age seconds"""
        with self._lock:
            now = time.time()
            expired_plates = [
                plate for plate, timestamp in self.recent_detections.items()
                if now - timestamp > max_age
            ]
            for plate in expired_plates:
                del self.recent_detections[plate]


class NotificationBatcher:
    """
    Group multiple rapid detections into single notification
    """
    def __init__(self, batch_timeout: float = 3.0, batch_size: int = 3):
        """
        Args:
            batch_timeout: Seconds to wait before flushing batch
            batch_size: Number of detections to trigger immediate flush
        """
        self.batch_timeout = batch_timeout
        self.batch_size = batch_size
        self.pending_batch: List[AlertEvent] = []
        self.last_batch_time = time.time()
        self._lock = threading.Lock()

    def add_detection(self, alert_event: AlertEvent) -> Tuple[bool, Optional[List[AlertEvent]]]:
        """
        Add detection to batch

        Returns:
            (should_flush, batch_events)
            - should_flush: True if batch should be sent now
            - batch_events: List of events if flushing, None otherwise
        """
        with self._lock:
            self.pending_batch.append(alert_event)

            now = time.time()
            time_elapsed = now - self.last_batch_time

            # Flush jika:
            # 1. Batch size reached
            # 2. Timeout elapsed
            # 3. CRITICAL priority detected
            should_flush = (
                len(self.pending_batch) >= self.batch_size or
                time_elapsed >= self.batch_timeout or
                alert_event.priority == AlertPriority.CRITICAL
            )

            if should_flush:
                batch = self.pending_batch.copy()
                self.pending_batch.clear()
                self.last_batch_time = now
                return True, batch

            return False, None

    def flush(self) -> Optional[List[AlertEvent]]:
        """Force flush current batch"""
        with self._lock:
            if self.pending_batch:
                batch = self.pending_batch.copy()
                self.pending_batch.clear()
                self.last_batch_time = time.time()
                return batch
            return None


class SoundManager:
    """
    Manage sound alerts dengan anti-spam
    """
    def __init__(self, min_sound_interval: float = 2.0):
        """
        Args:
            min_sound_interval: Minimum seconds between sounds
        """
        self.min_sound_interval = min_sound_interval
        self.last_sound_time = 0.0
        self.sound_queue: List[Tuple[str, AlertPriority]] = []
        self._lock = threading.Lock()

    def can_play_sound(self, priority: AlertPriority) -> bool:
        """
        Check if sound can be played now

        Args:
            priority: Alert priority

        Returns:
            True if sound can be played, False if too soon
        """
        with self._lock:
            now = time.time()
            time_since_last = now - self.last_sound_time

            # CRITICAL alerts always play
            if priority == AlertPriority.CRITICAL:
                self.last_sound_time = now
                return True

            # Others respect min interval
            if time_since_last >= self.min_sound_interval:
                self.last_sound_time = now
                return True

            return False

    def queue_sound(self, sound_type: str, priority: AlertPriority):
        """Add sound to queue"""
        with self._lock:
            self.sound_queue.append((sound_type, priority))

    def get_next_sound(self) -> Optional[Tuple[str, AlertPriority]]:
        """Get next sound dari queue"""
        with self._lock:
            if self.sound_queue:
                return self.sound_queue.pop(0)
            return None


class SmartAlertManager:
    """
    Main alert manager dengan session-based intelligence
    """
    def __init__(self):
        self.duplicate_filter = DuplicateDetectionFilter(cooldown_period=30)
        self.notification_batcher = NotificationBatcher(batch_timeout=3.0, batch_size=3)
        self.sound_manager = SoundManager(min_sound_interval=2.0)

        # Session statistics
        self.session_stats = {
            'total_detections': 0,
            'granted_count': 0,
            'denied_count': 0,
            'manual_override_count': 0,
            'start_time': time.time(),
            'last_activity': time.time()
        }

        # Activity tracking untuk busy hour detection
        self.activity_window: List[float] = []  # timestamps of detections
        self.busy_threshold = 20  # detections per hour

        self._lock = threading.Lock()

    def process_alert(
        self,
        alert_event: AlertEvent,
        user_settings: Optional[Dict] = None
    ) -> Dict:
        """
        Process alert dengan all anti-spam mechanisms

        Args:
            alert_event: Alert event to process
            user_settings: User alert preferences (dari database)

        Returns:
            Dict with processing result:
            {
                'should_show': bool,
                'should_sound': bool,
                'is_batched': bool,
                'batch_events': Optional[List],
                'sound_type': Optional[str],
                'reason': str  # Why decision was made
            }
        """
        with self._lock:
            # Update session stats
            self._update_session_stats(alert_event)

            # Default user settings
            if user_settings is None:
                user_settings = self._get_default_settings()

            result = {
                'should_show': False,
                'should_sound': False,
                'is_batched': False,
                'batch_events': None,
                'sound_type': None,
                'reason': ''
            }

            # 1. Check quiet hours
            if self._is_quiet_hours(user_settings) and alert_event.priority != AlertPriority.CRITICAL:
                result['reason'] = 'quiet_hours'
                return result

            # 2. Check DND mode
            if user_settings.get('enable_dnd', False) and alert_event.priority != AlertPriority.CRITICAL:
                result['reason'] = 'dnd_mode'
                return result

            # 3. Check priority filter
            if not self._should_show_priority(alert_event.priority, user_settings):
                result['reason'] = 'priority_filtered'
                return result

            # 4. Check duplicate (kecuali CRITICAL atau manual override)
            if alert_event.priority != AlertPriority.CRITICAL and not alert_event.manual_override:
                if not self.duplicate_filter.should_alert(alert_event.plate_number):
                    result['reason'] = 'duplicate_cooldown'
                    return result

            # 5. Check busy hour - show CRITICAL only
            if self._is_busy_hour() and alert_event.priority not in [AlertPriority.CRITICAL, AlertPriority.HIGH]:
                result['reason'] = 'busy_hour_filter'
                return result

            # 6. Batching (kecuali CRITICAL atau manual override)
            if alert_event.priority != AlertPriority.CRITICAL and not alert_event.manual_override:
                if user_settings.get('enable_grouping', True):
                    should_flush, batch = self.notification_batcher.add_detection(alert_event)
                    if should_flush and batch:
                        result['should_show'] = True
                        result['is_batched'] = True
                        result['batch_events'] = batch
                        result['reason'] = 'batched_flush'
                    else:
                        result['reason'] = 'batched_pending'
                        return result
            else:
                # CRITICAL and manual overrides bypass batching
                result['should_show'] = True
                result['reason'] = 'priority_bypass'

            # 7. Sound decision
            sound_enabled = self._should_play_sound(alert_event, user_settings)
            if sound_enabled:
                if self.sound_manager.can_play_sound(alert_event.priority):
                    result['should_sound'] = True
                    result['sound_type'] = self._get_sound_type(alert_event)
                else:
                    # Queue untuk nanti
                    self.sound_manager.queue_sound(
                        self._get_sound_type(alert_event),
                        alert_event.priority
                    )

            return result

    def _update_session_stats(self, alert_event: AlertEvent):
        """Update session statistics"""
        self.session_stats['total_detections'] += 1
        self.session_stats['last_activity'] = time.time()

        if alert_event.decision == 'granted':
            self.session_stats['granted_count'] += 1
        elif alert_event.decision == 'denied':
            self.session_stats['denied_count'] += 1

        if alert_event.manual_override:
            self.session_stats['manual_override_count'] += 1

        # Update activity window
        now = time.time()
        self.activity_window.append(now)

        # Keep only last hour
        one_hour_ago = now - 3600
        self.activity_window = [t for t in self.activity_window if t > one_hour_ago]

    def _is_busy_hour(self) -> bool:
        """Detect busy period based on activity rate"""
        if len(self.activity_window) < 5:
            return False

        # Calculate detections per hour
        now = time.time()
        one_hour_ago = now - 3600
        recent_detections = [t for t in self.activity_window if t > one_hour_ago]

        rate = len(recent_detections)
        return rate > self.busy_threshold

    def _is_quiet_hours(self, user_settings: Dict) -> bool:
        """Check if current time dalam quiet hours"""
        if not user_settings.get('enable_quiet_hours', False):
            return False

        now = datetime.now().time()
        start = user_settings.get('quiet_start_time', datetime.strptime('22:00', '%H:%M').time())
        end = user_settings.get('quiet_end_time', datetime.strptime('06:00', '%H:%M').time())

        if start <= end:
            return start <= now <= end
        else:  # Crosses midnight
            return now >= start or now <= end

    def _should_show_priority(self, priority: AlertPriority, user_settings: Dict) -> bool:
        """Check if priority should be shown based on user settings"""
        priority_map = {
            AlertPriority.CRITICAL: user_settings.get('show_critical', True),
            AlertPriority.HIGH: user_settings.get('show_high', True),
            AlertPriority.MEDIUM: user_settings.get('show_medium', False),
            AlertPriority.LOW: user_settings.get('show_low', False)
        }
        return priority_map.get(priority, False)

    def _should_play_sound(self, alert_event: AlertEvent, user_settings: Dict) -> bool:
        """Determine if sound should be played"""
        if not user_settings.get('enable_audio', True):
            return False

        # Check sound settings per type
        if alert_event.decision == 'denied':
            return user_settings.get('sound_denied', True)
        elif alert_event.decision == 'granted' and alert_event.manual_override:
            return user_settings.get('sound_granted_manual', True)
        elif alert_event.decision == 'granted':
            return user_settings.get('sound_granted_auto', False)
        elif alert_event.decision == 'pending':
            return user_settings.get('sound_manual_required', True)

        return False

    def _get_sound_type(self, alert_event: AlertEvent) -> str:
        """Get appropriate sound file name"""
        if alert_event.decision == 'granted':
            return 'access_granted.mp3'
        elif alert_event.decision == 'denied':
            return 'access_denied.mp3'
        elif alert_event.decision == 'pending':
            return 'manual_required.mp3'
        elif alert_event.manual_override:
            return 'manual_override.mp3'
        return 'default.mp3'

    def _get_default_settings(self) -> Dict:
        """Get default alert settings"""
        return {
            'enable_audio': True,
            'audio_volume': 0.8,
            'sound_denied': True,
            'sound_granted_auto': False,
            'sound_granted_manual': True,
            'sound_manual_required': True,
            'auto_dismiss_seconds': 5,
            'max_visible_alerts': 3,
            'enable_grouping': True,
            'show_critical': True,
            'show_high': True,
            'show_medium': False,
            'show_low': False,
            'enable_quiet_hours': False,
            'quiet_start_time': datetime.strptime('22:00', '%H:%M').time(),
            'quiet_end_time': datetime.strptime('06:00', '%H:%M').time(),
            'enable_dnd': False
        }

    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        with self._lock:
            uptime = time.time() - self.session_stats['start_time']
            return {
                **self.session_stats,
                'uptime_seconds': uptime,
                'detections_per_hour': len(self.activity_window),
                'is_busy': self._is_busy_hour()
            }

    def reset_session(self):
        """Reset session statistics"""
        with self._lock:
            self.session_stats = {
                'total_detections': 0,
                'granted_count': 0,
                'denied_count': 0,
                'manual_override_count': 0,
                'start_time': time.time(),
                'last_activity': time.time()
            }
            self.activity_window.clear()
            self.duplicate_filter.recent_detections.clear()


# Global singleton instance
_alert_manager_instance = None

def get_alert_manager() -> SmartAlertManager:
    """Get global SmartAlertManager instance"""
    global _alert_manager_instance
    if _alert_manager_instance is None:
        _alert_manager_instance = SmartAlertManager()
    return _alert_manager_instance
