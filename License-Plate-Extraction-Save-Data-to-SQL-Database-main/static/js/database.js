document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const typeFilter = document.getElementById('type-filter');
    const deleteButtons = document.querySelectorAll('.delete-plate');

    // Search functionality
    searchBtn.addEventListener('click', () => {
        const searchTerm = searchInput.value.toLowerCase();
        const rows = document.querySelectorAll('tbody tr');

        rows.forEach(row => {
            const plateNumber = row.cells[0].textContent.toLowerCase();
            const plateType = row.cells[1].textContent.toLowerCase();
            const matchesSearch = plateNumber.includes(searchTerm);
            const matchesType = !typeFilter.value || plateType === typeFilter.value.toLowerCase();

            row.style.display = matchesSearch && matchesType ? '' : 'none';
        });
    });

    // Type filter functionality
    typeFilter.addEventListener('change', () => {
        searchBtn.click(); // Trigger search with current filters
    });

    // Delete plate functionality
    deleteButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const plateId = button.dataset.id;
            if (confirm('Are you sure you want to delete this license plate?')) {
                try {
                    const response = await fetch(`/api/plates/${plateId}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        button.closest('tr').remove();
                    } else {
                        const data = await response.json();
                        throw new Error(data.error || 'Failed to delete plate');
                    }
                } catch (error) {
                    alert(`Error: ${error.message}`);
                }
            }
        });
    });

    // Real-time search
    searchInput.addEventListener('input', () => {
        searchBtn.click();
    });
});
