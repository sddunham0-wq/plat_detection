/**
 * Vehicles Management JavaScript
 * Handles DataTables initialization and delete operations
 */

$(document).ready(function() {
    // Initialize DataTables
    const table = $('#vehiclesTable').DataTable({
        "order": [[0, "asc"]], // Sort by plate number
        "pageLength": 25,
        "responsive": true,
        "language": {
            "search": "Search:",
            "lengthMenu": "Show _MENU_ vehicles per page",
            "info": "Showing _START_ to _END_ of _TOTAL_ vehicles",
            "infoEmpty": "No vehicles found",
            "infoFiltered": "(filtered from _MAX_ total vehicles)",
            "zeroRecords": "No matching vehicles found",
            "paginate": {
                "first": "First",
                "last": "Last",
                "next": "Next",
                "previous": "Previous"
            }
        },
        "columnDefs": [
            {
                "targets": 6, // Actions column
                "orderable": false,
                "searchable": false
            }
        ]
    });

    // Delete button click handler
    let deleteVehicleId = null;

    $(document).on('click', '.btn-delete', function() {
        deleteVehicleId = $(this).data('id');
        const plateNumber = $(this).data('plate');

        // Set plate number in modal
        $('#deletePlateNumber').text(plateNumber);
        $('#deletePin').val('');

        // Show modal
        const deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'));
        deleteModal.show();
    });

    // Confirm delete button
    $('#confirmDelete').click(function() {
        const pin = $('#deletePin').val();

        // Validate PIN (simple check, default: 1234)
        if (pin !== 'cctv1234') {
            showToast('Invalid PIN! Default PIN is cctv1234', 'danger');
            return;
        }

        // Send delete request
        $.ajax({
            url: `/vehicles/delete/${deleteVehicleId}`,
            method: 'POST',
            data: {
                pin: pin
            },
            success: function(response) {
                if (response.success) {
                    showToast(response.message, 'success');

                    // Hide modal
                    bootstrap.Modal.getInstance(document.getElementById('deleteModal')).hide();

                    // Reload page after 1 second
                    setTimeout(function() {
                        window.location.reload();
                    }, 1000);
                } else {
                    showToast(response.message || 'Failed to delete vehicle', 'danger');
                }
            },
            error: function(xhr) {
                const response = xhr.responseJSON;
                showToast(response?.message || 'Error deleting vehicle', 'danger');
            }
        });
    });

    // Clear PIN when modal is hidden
    $('#deleteModal').on('hidden.bs.modal', function() {
        $('#deletePin').val('');
        deleteVehicleId = null;
    });

    // Auto-refresh statistics every 30 seconds
    setInterval(function() {
        $.ajax({
            url: '/api/vehicles/stats',
            method: 'GET',
            success: function(data) {
                if (data) {
                    $('#totalVehiclesCount').text(data.total || 0);
                    $('#presentCount').text(data.present || 0);
                    $('#absentCount').text(data.absent || 0);
                    $('#todayAccessCount').text(data.today_access || 0);
                }
            },
            error: function() {
                console.error('Failed to fetch statistics');
            }
        });
    }, 30000);
});
