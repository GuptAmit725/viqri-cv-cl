// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadContent = document.getElementById('uploadContent');
const uploadProgress = document.getElementById('uploadProgress');
const uploadSuccess = document.getElementById('uploadSuccess');
const fileInfo = document.getElementById('fileInfo');
const submitBtn = document.getElementById('submitBtn');
const progressFill = document.getElementById('progressFill');
const progressPercentage = document.getElementById('progressPercentage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const successFilename = document.getElementById('successFilename');
const removeFileBtn = document.getElementById('removeFile');

// Sections
const uploadSection = document.getElementById('uploadSection');
const jobDetailsSection = document.getElementById('jobDetailsSection');
const resultsSection = document.getElementById('resultsSection');

// Form elements
const jobForm = document.getElementById('jobForm');
const targetJobInput = document.getElementById('targetJob');
const targetLocationInput = document.getElementById('targetLocation');
const industrySelect = document.getElementById('industry');
const experienceLevelSelect = document.getElementById('experienceLevel');

// Results
const resultsLoading = document.getElementById('resultsLoading');
const resultsContent = document.getElementById('resultsContent');
const startOverBtn = document.getElementById('startOverBtn');

let selectedFile = null;
let cvData = null;

// API Configuration
const API_URL = 'https://viqri-cv-api-5u7hdc64va-uc.a.run.app';

// Initialize
init();

function init() {
    setupEventListeners();
}

function setupEventListeners() {
    // Upload area
    uploadArea.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    submitBtn.addEventListener('click', handleCVAnalysis);
    
    // Job form
    jobForm.addEventListener('submit', handleTemplateGeneration);
    
    // Start over
    startOverBtn.addEventListener('click', startOver);

    // Prevent default drag behavior
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
}

function handleFileSelect(file) {
    if (!file) return;

    const validTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a PDF, DOC, or DOCX file', 'error');
        return;
    }

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File size must be less than 10MB', 'error');
        return;
    }

    selectedFile = file;
    displayFileInfo(file);
    showSubmitButton();
}

function displayFileInfo(file) {
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
    const sizeInKB = (file.size / 1024).toFixed(2);
    const displaySize = file.size > 1024 * 1024 ? `${sizeInMB} MB` : `${sizeInKB} KB`;

    fileName.textContent = file.name;
    fileSize.textContent = displaySize;

    fileInfo.style.display = 'block';
    uploadArea.style.cursor = 'default';
}

function showSubmitButton() {
    submitBtn.style.display = 'flex';
}

function hideSubmitButton() {
    submitBtn.style.display = 'none';
}

async function handleCVAnalysis() {
    if (!selectedFile) return;

    fileInfo.style.display = 'none';
    hideSubmitButton();

    uploadContent.style.display = 'none';
    uploadProgress.style.display = 'flex';

    await uploadAndParseCV();
}

async function uploadAndParseCV() {
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 90) {
                progress = 90;
            }
            updateProgress(progress);
        }, 200);

        const response = await fetch(`${API_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        updateProgress(100);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
        }

        const result = await response.json();

        if (result.success) {
            cvData = result.data;
            console.log('✅ CV Parsed Successfully!', cvData);
            
            setTimeout(() => {
                showCVSuccess();
                showJobDetailsForm();
            }, 500);
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('❌ Upload error:', error);
        showNotification('Failed to analyze CV: ' + error.message, 'error');
        resetUpload();
    }
}

function showCVSuccess() {
    uploadProgress.style.display = 'none';
    uploadSuccess.style.display = 'flex';
    successFilename.textContent = selectedFile.name;
    showNotification('CV analyzed successfully! Now let\'s optimize it.', 'success');
}

function showJobDetailsForm() {
    setTimeout(() => {
        jobDetailsSection.style.display = 'block';
        jobDetailsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 1000);
}

async function handleTemplateGeneration(e) {
    e.preventDefault();

    const targetJob = targetJobInput.value.trim();
    const targetLocation = targetLocationInput.value.trim();
    const industry = industrySelect.value;
    const experienceLevel = experienceLevelSelect.value;

    if (!targetJob || !targetLocation) {
        showNotification('Please fill in required fields', 'error');
        return;
    }

    // Show results section with loading
    resultsSection.style.display = 'block';
    resultsLoading.style.display = 'block';
    resultsContent.style.display = 'none';
    
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
        const response = await fetch(`${API_URL}/api/generate-template`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cv_data: cvData,
                target_job: targetJob,
                target_location: targetLocation,
                industry: industry || null,
                experience_level: experienceLevel || null
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Template generation failed');
        }

        const result = await response.json();

        if (result.success) {
            console.log('✅ Template Generated!', result.data);
            
            // Save data to localStorage
            localStorage.setItem('templateData', JSON.stringify(result.data));
            localStorage.setItem('cvData', JSON.stringify(cvData));
            localStorage.setItem('jobDetails', JSON.stringify({
                target_job: targetJob,
                target_location: targetLocation,
                industry: industry,
                experience_level: experienceLevel
            }));
            
            // Show success message
            showNotification('Template generated! Redirecting...', 'success');
            
            // Redirect to template page
            setTimeout(() => {
                window.location.href = 'template.html';
            }, 1500);
        } else {
            throw new Error('Template generation failed');
        }

    } catch (error) {
        console.error('❌ Template generation error:', error);
        showNotification('Failed to generate template: ' + error.message, 'error');
        resultsLoading.style.display = 'none';
    }
}

function displayResults(templateData) {
    resultsLoading.style.display = 'none';
    resultsContent.style.display = 'block';

    // Create results HTML
    let html = '<div class="results-grid">';

    // Template Structure
    if (templateData.template_structure) {
        html += `
            <div class="result-card">
                <h3>📋 Recommended Structure</h3>
                <div class="result-content">
                    <p><strong>Format:</strong> ${templateData.template_structure.format || 'N/A'}</p>
                    <p><strong>Length:</strong> ${templateData.template_structure.length || 'N/A'}</p>
                    <p><strong>Sections:</strong></p>
                    <ul>
                        ${(templateData.template_structure.sections || []).map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }

    // Content Recommendations
    if (templateData.content_recommendations) {
        const rec = templateData.content_recommendations;
        html += `
            <div class="result-card">
                <h3>✨ Content Recommendations</h3>
                <div class="result-content">
                    ${rec.summary ? `<p><strong>Professional Summary:</strong> ${rec.summary}</p>` : ''}
                    ${rec.key_skills ? `
                        <p><strong>Key Skills to Highlight:</strong></p>
                        <div class="skills-tags">
                            ${rec.key_skills.map(s => `<span class="skill-tag">${s}</span>`).join('')}
                        </div>
                    ` : ''}
                    ${rec.missing_skills && rec.missing_skills.length > 0 ? `
                        <p><strong>⚠️ Missing Skills:</strong></p>
                        <ul class="warning-list">
                            ${rec.missing_skills.map(s => `<li>${s}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            </div>
        `;
    }

    // Location Specific
    if (templateData.location_specific) {
        const loc = templateData.location_specific;
        html += `
            <div class="result-card">
                <h3>🌍 Location Insights</h3>
                <div class="result-content">
                    ${loc.format_preferences ? `<p><strong>Format:</strong> ${loc.format_preferences}</p>` : ''}
                    ${loc.cultural_considerations ? `
                        <p><strong>Cultural Considerations:</strong></p>
                        <ul>
                            ${loc.cultural_considerations.map(c => `<li>${c}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            </div>
        `;
    }

    // Action Items
    if (templateData.action_items) {
        const actions = templateData.action_items;
        html += `
            <div class="result-card action-card">
                <h3>🎯 Action Items</h3>
                <div class="result-content">
                    ${actions.immediate ? `
                        <div class="action-section priority-high">
                            <h4>🔥 Do This Now:</h4>
                            <ul>
                                ${actions.immediate.map(a => `<li>${a}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${actions.important ? `
                        <div class="action-section priority-medium">
                            <h4>⚡ Important Changes:</h4>
                            <ul>
                                ${actions.important.map(a => `<li>${a}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    html += '</div>';

    // Display raw data in console
    console.log('📊 Full Template Data:', templateData);

    resultsContent.innerHTML = html;
    showNotification('Template generated successfully!', 'success');
}

function updateProgress(progress) {
    const percent = Math.min(Math.round(progress), 100);
    progressFill.style.width = `${percent}%`;
    progressPercentage.textContent = `${percent}%`;
}

function resetUpload() {
    selectedFile = null;
    cvData = null;
    fileInput.value = '';

    uploadContent.style.display = 'block';
    uploadProgress.style.display = 'none';
    uploadSuccess.style.display = 'none';
    fileInfo.style.display = 'none';
    hideSubmitButton();
    
    uploadArea.style.cursor = 'pointer';
    progressFill.style.width = '0%';
    progressPercentage.textContent = '0%';
}

function startOver() {
    resetUpload();
    jobDetailsSection.style.display = 'none';
    resultsSection.style.display = 'none';
    jobForm.reset();
    
    uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    Object.assign(notification.style, {
        position: 'fixed',
        top: '2rem',
        right: '2rem',
        padding: '1rem 1.5rem',
        borderRadius: '12px',
        background: type === 'success' ? '#48bb78' : type === 'error' ? '#f56565' : '#4299e1',
        color: 'white',
        fontWeight: '600',
        fontSize: '0.875rem',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
        zIndex: '1000',
        animation: 'slideInRight 0.3s ease-out',
        maxWidth: '320px'
    });

    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(400px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(400px); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

document.documentElement.style.scrollBehavior = 'smooth';

window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease-in';
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

console.log('🚀 Viqri CV - AI Template Generator Ready!');
console.log('🔗 Backend API:', API_URL);
console.log('🤖 Powered by Groq Llama 3.1');