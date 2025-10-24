/**
 * Access Log JavaScript
 * Handles filtering, DataTables, and CSV export
 */

$(document).ready(function() {
    // Initialize DataTables
    const table = $('#accessLogTable').DataTable({
        "order": [[1, "desc"]], // Sort by timestamp descending (newest first) - column index shifted due to checkbox
        "pageLength": 50,
        "responsive": true,
        "language": {
            "search": "Search:",
            "lengthMenu": "Show _MENU_ records per page",
            "info": "Showing _START_ to _END_ of _TOTAL_ records",
            "infoEmpty": "No records found",
            "infoFiltered": "(filtered from _MAX_ total records)",
            "zeroRecords": "No matching records found",
            "paginate": {
                "first": "First",
                "last": "Last",
                "next": "Next",
                "previous": "Previous"
            }
        },
        "columnDefs": [
            {
                "targets": 0, // Checkbox column
                "orderable": false,
                "searchable": false
            },
            {
                "targets": 6, // Image column
                "orderable": false,
                "searchable": false
            }
        ]
    });

    // Date range selector
    $('#dateRange').change(function() {
        const value = $(this).val();

        if (value === 'custom') {
            $('#customDateStart').show();
            $('#customDateEnd').show();
        } else {
            $('#customDateStart').hide();
            $('#customDateEnd').hide();
        }
    });

    // Auto-uppercase plate search
    $('#plateSearch').on('input', function() {
        $(this).val($(this).val().toUpperCase());
    });

    // Filter form submission
    $('#filterForm').submit(function(e) {
        e.preventDefault();

        const formData = {
            dateRange: $('#dateRange').val(),
            startDate: $('#startDate').val(),
            endDate: $('#endDate').val(),
            status: $('#statusFilter').val(),
            plateSearch: $('#plateSearch').val()
        };

        // Reload page with filters
        const params = new URLSearchParams();

        if (formData.dateRange === 'custom') {
            if (formData.startDate) params.append('start_date', formData.startDate);
            if (formData.endDate) params.append('end_date', formData.endDate);
        } else {
            params.append('date_range', formData.dateRange);
        }

        if (formData.status) params.append('status', formData.status);
        if (formData.plateSearch) params.append('plate', formData.plateSearch);

        window.location.href = `/access-log?${params.toString()}`;
    });

    // CSV Export
    $('#exportCsvBtn').click(function() {
        const formData = {
            dateRange: $('#dateRange').val(),
            startDate: $('#startDate').val(),
            endDate: $('#endDate').val(),
            status: $('#statusFilter').val(),
            plateSearch: $('#plateSearch').val()
        };

        // Build export URL with filters
        const params = new URLSearchParams();

        if (formData.dateRange === 'custom') {
            if (formData.startDate) params.append('start_date', formData.startDate);
            if (formData.endDate) params.append('end_date', formData.endDate);
        } else {
            params.append('date_range', formData.dateRange);
        }

        if (formData.status) params.append('status', formData.status);
        if (formData.plateSearch) params.append('plate', formData.plateSearch);

        // Download CSV
        window.location.href = `/access-log/export?${params.toString()}`;
    });

    // View image button
    $(document).on('click', '.view-image-btn', function() {
        const imagePath = $(this).data('image');
        const plateNumber = $(this).data('plate');

        // Set modal content
        $('#modalPlateNumber').text(plateNumber);
        $('#modalImage').attr('src', '/' + imagePath);

        // Show modal
        const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
        imageModal.show();
    });

    // Update statistics via AJAX (if needed)
    function updateStatistics() {
        $.ajax({
            url: '/api/access-log/stats',
            method: 'GET',
            data: {
                date_range: $('#dateRange').val(),
                status: $('#statusFilter').val(),
                plate: $('#plateSearch').val()
            },
            success: function(data) {
                if (data) {
                    $('#totalRecords').text(data.total || 0);
                    $('#entryCount').text(data.entry || 0);
                    $('#exitCount').text(data.exit || 0);
                    $('#deniedCount').text(data.denied || 0);
                }
            },
            error: function() {
                console.error('Failed to fetch access log statistics');
            }
        });
    }

    // Auto-refresh statistics every 30 seconds
    setInterval(updateStatistics, 30000);

    // ========== BULK DELETE FUNCTIONALITY ==========

    let selectedIds = [];

    // Select All checkbox
    $('#selectAll').change(function() {
        const isChecked = $(this).is(':checked');
        $('.log-checkbox').prop('checked', isChecked);
        updateSelectedCount();
    });

    // Individual checkbox
    $(document).on('change', '.log-checkbox', function() {
        updateSelectedCount();

        // Update "Select All" state
        const totalCheckboxes = $('.log-checkbox').length;
        const checkedCheckboxes = $('.log-checkbox:checked').length;
        $('#selectAll').prop('checked', totalCheckboxes === checkedCheckboxes);
    });

    // Update selected count and show/hide delete button
    function updateSelectedCount() {
        selectedIds = [];
        $('.log-checkbox:checked').each(function() {
            selectedIds.push($(this).val());
        });

        const count = selectedIds.length;
        $('#selectedCount').text(count);

        if (count > 0) {
            $('#bulkDeleteBtn').show();
        } else {
            $('#bulkDeleteBtn').hide();
        }
    }

    // Bulk delete button click
    $('#bulkDeleteBtn').click(function() {
        if (selectedIds.length === 0) {
            showToast('Please select at least one record to delete', 'warning');
            return;
        }

        $('#deleteCount').text(selectedIds.length);
        $('#bulkDeletePin').val('');

        const bulkDeleteModal = new bootstrap.Modal(document.getElementById('bulkDeleteModal'));
        bulkDeleteModal.show();
    });

    // Confirm bulk delete
    $('#confirmBulkDelete').click(function() {
        const pin = $('#bulkDeletePin').val();

        if (pin !== 'cctv1234') {
            showToast('Invalid PIN!', 'danger');
            return;
        }

        if (selectedIds.length === 0) {
            showToast('No records selected', 'warning');
            return;
        }

        // Disable button and show loading
        const $btn = $(this);
        const originalHTML = $btn.html();
        $btn.html('<span class="spinner-border spinner-border-sm me-1"></span>Deleting...').prop('disabled', true);

        // Send delete request
        $.ajax({
            url: '/access-log/bulk-delete',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                ids: selectedIds,
                pin: pin
            }),
            success: function(response) {
                if (response.success) {
                    showToast(`Successfully deleted ${response.deleted_count} record(s)`, 'success');

                    // Close modal
                    bootstrap.Modal.getInstance(document.getElementById('bulkDeleteModal')).hide();

                    // Reload page after 1 second
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } else {
                    showToast('Failed to delete records: ' + response.error, 'danger');
                    $btn.html(originalHTML).prop('disabled', false);
                }
            },
            error: function(xhr) {
                console.error('Delete error:', xhr);
                showToast('Network error while deleting records', 'danger');
                $btn.html(originalHTML).prop('disabled', false);
            }
        });
    });

    // Toast notification helper
    function showToast(message, type) {
        const bgClass = type === 'success' ? 'bg-success' :
                       type === 'danger' ? 'bg-danger' :
                       type === 'warning' ? 'bg-warning' : 'bg-info';

        const toast = $(`
            <div class="toast align-items-center text-white ${bgClass} border-0" role="alert" style="position: fixed; top: 20px; right: 20px; z-index: 9999;">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `);

        $('body').append(toast);
        const bsToast = new bootstrap.Toast(toast[0]);
        bsToast.show();

        // Remove from DOM after hidden
        toast.on('hidden.bs.toast', function() {
            $(this).remove();
        });
    }
});
