        alert("SCRIPT LOADED");
        
        const API_URL = 'http://localhost:8000';
        let predictionCount = 0;
        let totalConfidence = 0;

        const examples = {
            billing: {
                subject: "Charged twice for subscription",
                description: "I was charged $49.99 twice on my credit card this month for my premium subscription. I only have one account and should only be charged once. Please investigate and refund the duplicate charge."
            },
            technical: {
                subject: "Application crashes on startup",
                description: "The application crashes immediately after launching on Windows 11. I've tried reinstalling but the problem persists. Error code: 0x80004005. This started happening after the latest update."
            },
            account: {
                subject: "Cannot login to my account",
                description: "I've been trying to login for an hour but keep getting 'Invalid credentials' error. I'm sure my password is correct. I tried resetting it but the reset email never arrives."
            },
            feature: {
                subject: "Request for dark mode",
                description: "Would love to see a dark mode option in the app. Many of us work late at night and the bright interface is hard on the eyes. This would greatly improve the user experience."
            },
            general: {
                subject: "Question about pricing plans",
                description: "I'm interested in upgrading from the free plan but confused about the differences between Pro and Enterprise tiers. Can you explain what features are included in each?"
            }
        };

        const categoryClasses = {
            'Billing': 'category-billing',
            'Technical': 'category-technical',
            'Account': 'category-account',
            'Feature Request': 'category-feature',
            'General Inquiry': 'category-general'
        };

        const priorityClasses = {
            'Critical': 'priority-critical',
            'High': 'priority-high',
            'Medium': 'priority-medium',
            'Low': 'priority-low'
        };

        document.getElementById('ticketForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            await classifyTicket();
        });

        async function classifyTicket() {
            const subject = document.getElementById('subject').value;
            const description = document.getElementById('description').value;

            if (!subject || !description) {
                showToast('Please fill in both subject and description');
                return;
            }

            // Show loading
            const btn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');

            btn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'block';

            const startTime = performance.now();

            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ subject, description })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                const endTime = performance.now();
                const responseTime = Math.round(endTime - startTime);

                displayResults(data, responseTime);

            } catch (error) {
                console.error('Error:', error);
                showToast('Failed to classify ticket. Is the API running?');
            } finally {
                btn.disabled = false;
                btnText.style.display = 'inline';
                spinner.style.display = 'none';
            }
        }

        function displayResults(data, responseTime) {
            const results = document.getElementById('results');
            results.classList.add('show');

            // Update category
            const categoryBadge = document.getElementById('categoryBadge');
            categoryBadge.textContent = data.category;
            categoryBadge.className = 'result-badge ' + (categoryClasses[data.category] || '');

            const catConf = Math.round(data.confidence_category * 100);
            document.getElementById('categoryConfidence').style.width = catConf + '%';
            document.getElementById('categoryConfidenceText').textContent = `Confidence: ${catConf}%`;

            // Update priority
            const priorityBadge = document.getElementById('priorityBadge');
            priorityBadge.textContent = data.priority;
            priorityBadge.className = 'result-badge ' + (priorityClasses[data.priority] || '');

            const priConf = Math.round(data.confidence_priority * 100);
            document.getElementById('priorityConfidence').style.width = priConf + '%';
            document.getElementById('priorityConfidenceText').textContent = `Confidence: ${priConf}%`;

            // Update probability lists
            updateProbList('categoryProbs', data.category_probabilities);
            updateProbList('priorityProbs', data.priority_probabilities);

            // Update stats
            
            predictionCount++;
            totalConfidence += (data.confidence_category + data.confidence_priority) / 2;

            document.getElementById('totalPredictions').textContent = predictionCount;
            document.getElementById('avgConfidence').textContent = Math.round(totalConfidence / predictionCount * 100) + '%';
            document.getElementById('responseTime').textContent = responseTime + 'ms';

            predictionHistory.unshift({
                category: data.category,
                priority: data.priority,
                confidence: Math.round(
                    data.confidence_category * 100
                ),
                timestamp: Date.now()
            });

            predictionHistory =
                predictionHistory.slice(0, 5);

            localStorage.setItem(
                'predictionHistory',
                JSON.stringify(predictionHistory)
            );

            renderHistory();
        }

        function updateProbList(elementId, probs) {
            const container = document.getElementById(elementId);
            container.innerHTML = '';

            // Sort by probability
            const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

            sorted.forEach(([name, prob]) => {
                const percentage = Math.round(prob * 100);
                container.innerHTML += `
                    <div class="prob-item">
                        <span class="prob-name">${name}</span>
                        <div class="prob-bar-wrap">
                            <div class="prob-bar" style="width: ${percentage}%"></div>
                        </div>
                        <span class="prob-value">${percentage}%</span>
                    </div>
                `;
            });
        }

        function toggleProbs(type) {
            const list = document.getElementById(type + 'Probs');
            const arrow = document.getElementById(type + 'Arrow');

            if (list.classList.contains('show')) {
                list.classList.remove('show');
                arrow.textContent = '▶';
            } else {
                list.classList.add('show');
                arrow.textContent = '▼';
            }
        }

        function loadExample(type) {
            const example = examples[type];
            if (example) {
                document.getElementById('subject').value = example.subject;
                document.getElementById('description').value = example.description;
            }
        }

        function loadRandomExample() {
            const types = Object.keys(examples);
            const randomType = types[Math.floor(Math.random() * types.length)];
            loadExample(randomType);
        }

        function clearForm() {
            document.getElementById('subject').value = '';
            document.getElementById('description').value = '';
            document.getElementById('results').classList.remove('show');
        }

        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');

            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        }

        function clearHistory() {

            predictionHistory = [];
            localStorage.removeItem(
                'predictionHistory'
            );
            renderHistory();
            showToast('History cleared');
        }

        // Check API health on load
        async function checkHealth() {
            try {
                const response = await fetch(`${API_URL}/health`);
                const data = await response.json();
                if (!data.model_loaded) {
                    showToast('⚠️ Model not loaded. Train the model first.');
                }
            } catch (error) {
                showToast('⚠️ API not reachable. Start the server.');
            }
        }

        let predictionHistory =
        JSON.parse(
            localStorage.getItem('predictionHistory')
        ) || [];

        function renderHistory() {

        const container =
            document.getElementById('historyList');

        if (!predictionHistory.length) {

            container.innerHTML =
                '<p class="history-empty">No predictions yet</p>';

            return;
        }

container.innerHTML =
    predictionHistory
        .map(item => {

            const minutes =
                Math.floor(
                    (Date.now() - item.timestamp)
                    / 60000
                );

            const timeText =
                minutes < 1
                    ? 'Just now'
                    : `${minutes} min ago`;

            return `
                <div class="history-item">

                    <div class="history-main">

                        <div class="history-top">

                            <span class="history-category">
                                ${item.category}
                            </span>

                            <span class="history-priority-chip">
                                ${item.priority}
                            </span>

                        </div>

                        <span class="history-time">
                            ${timeText}
                        </span>

                    </div>

                    <span class="history-confidence">
                        ${item.confidence}%
                    </span>

                </div>
            `;
        })
        .join('');
    }
        renderHistory();
        checkHealth();
