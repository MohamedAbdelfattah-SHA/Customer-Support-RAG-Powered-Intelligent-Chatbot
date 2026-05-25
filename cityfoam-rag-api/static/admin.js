document.addEventListener('DOMContentLoaded', function () {
    // عناصر الواجهة
    //added this so the text chamges whn we select a file
    document.getElementById('fileInput').addEventListener('change', function() {
        const fileName = this.files[0] ? this.files[0].name : "No file selected...";
        document.getElementById('fileNameDisplay').textContent = fileName;
        document.getElementById('fileNameDisplay').style.color = "#0f172a";
        document.getElementById('fileNameDisplay').style.fontWeight = "bold";
    });

    const uploadBtn = document.getElementById('uploadBtn');
    const retrainBtn = document.getElementById('retrainBtn');
    const statusBox = document.getElementById('statusBox');
    const fileInput = document.getElementById('fileInput');

    // --- رفع الملف ---
    //new upload file logic!! cos' the upload part didn't come first
    uploadBtn.addEventListener('click', async () => {
        if (!fileInput.files.length) {
            alert('يرجى اختيار ملف أولاً! (Please select a file first!)');
            return;
        }

        // 1. Create the formData object FIRST
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        // 2. Get the security token
        const token = sessionStorage.getItem('admin_token');

        // 3. UI Loading State
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';

        try {
            // 4. Send file to Python backend (formData is safely used here)
            const response = await fetch('/api/upload-knowledge', {
                method: 'POST',
                headers: {
                    'X-API-Key': token
                },
                body: formData 
            });
            
            const data = await response.json();

            if (response.ok) {
                alert('نجاح: ' + data.message);
                
                // Reset the UI after successful upload
                fileInput.value = ''; 
                document.getElementById('fileNameDisplay').textContent = "No file selected...";
                document.getElementById('fileNameDisplay').style.fontWeight = "normal";
                document.getElementById('fileNameDisplay').style.color = "#64748b";
                
                statusBox.textContent = "File uploaded! Ready for retraining.";
                statusBox.style.color = "#4f46e5";
            } else {
                alert('فشل الرفع: ' + (data.detail || data.message));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('حدث خطأ في الاتصال بالسيرفر.');
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = '🚀 2. Upload to Server';
        }
    });

    // --- إعادة التدريب ---
    //ordered retraining logic only
    retrainBtn.addEventListener('click', async () => {
        // 1. Grab the security token FIRST
        const token = sessionStorage.getItem('admin_token');

        if (!token) {
            alert("You are not logged in! Please refresh the page and log in again.");
            return;
        }

        // 2. UI Loading State
        statusBox.style.color = '#eab308'; // Warning Yellow
        statusBox.textContent = 'Status: Retraining in progress... This may take a few minutes as the AI reads the new files.';
        retrainBtn.disabled = true;

        try {
            // 3. Trigger Python Retraining Pipeline with the Security Token
            const response = await fetch('/api/trigger-retraining', {
                method: 'POST',
                headers: {
                    'X-API-Key': token // <-- THIS IS THE VIP PASS THAT WAS MISSING
                }
            });
            
            const data = await response.json();

            // 4. Handle the server's response
            if (response.ok && data.status !== 'error') {
                if (data.status === 'idle') {
                    statusBox.style.color = '#3b82f6'; // Blue
                    statusBox.textContent = 'Status: ' + data.message;
                } else if (data.status === 'success') {
                    statusBox.style.color = '#22c55e'; // Green
                    statusBox.textContent = 'Status: Success! ' + data.message;
                    updateMetricsTable(data.metrics);
                } else if (data.status === 'failed') {
                    statusBox.style.color = '#ef4444'; // Red
                    statusBox.textContent = 'Status: Failed (Metrics Degraded). Database reverted to safe backup.';
                    updateMetricsTable(data.metrics);
                }
            } else {
                // If the server returns a 500 or 401 error
                statusBox.style.color = '#ef4444';
                statusBox.textContent = 'Status: Error - ' + (data.message || data.detail || 'Unknown server error');
            }
        } catch (error) {
            console.error('Error:', error);
            statusBox.style.color = '#ef4444';
            statusBox.textContent = 'Status: Connection error during retraining.';
        } finally {
            // Reset the button
            retrainBtn.disabled = false;
        }
    });

    // --- تحديث الجدول ---
    function updateMetricsTable(metricsArray) {
        const rows = document.querySelectorAll('#metricsTable tr');

        metricsArray.forEach(function (item, index) {
            const incomingMetricName = item.metric.trim().toUpperCase();
            let matchedRow = null;

            rows.forEach(function (row) {
                const rowMetricName = row.cells[0].innerText.trim().toUpperCase();
                if (rowMetricName.includes(incomingMetricName) || incomingMetricName.includes(rowMetricName)) {
                    matchedRow = row;
                }
            });

            if (!matchedRow && rows[index]) {
                matchedRow = rows[index];
            }

            if (matchedRow) {
                const unit = incomingMetricName.includes('LATENCY') ? 's' : '%';

                matchedRow.cells[1].innerText = item.previous + unit;
                matchedRow.cells[2].innerText = item.current + unit;

                const changeCell = matchedRow.cells[3];
                changeCell.innerText = (item.change >= 0 ? '+' : '') + item.change + unit;
                changeCell.style.fontWeight = 'bold';

                if (incomingMetricName.includes('HALLUCINATION') || incomingMetricName.includes('LATENCY')) {
                    if (item.change > 0) changeCell.style.color = 'red';
                    else if (item.change < 0) changeCell.style.color = 'green';
                } else {
                    if (item.change > 0) changeCell.style.color = 'green';
                    else if (item.change < 0) changeCell.style.color = 'red';
                }
            }
        });
    }
});