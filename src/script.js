// ============================================
// Viqri CV - Modern JavaScript Controller
// ============================================

// State Management
const state = {
    currentStep: 1,
    uploadedFile: null,
    cvData: null,
    jobDetails: null,
    initialized: false
};

// DOM Elements
const elements = {
    // Upload Section
    uploadSection: document.getElementById('uploadSection'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    uploadIdle: document.getElementById('uploadIdle'),
    uploadProcessing: document.getElementById('uploadProcessing'),
    uploadSuccess: document.getElementById('uploadSuccess'),
    progressBar: document.getElementById('progressBar'),
    successFilename: document.getElementById('successFilename'),
    changeFile: document.getElementById('changeFile'),
    continueBtn: document.getElementById('continueBtn'),
    
    // Job Details Section
    jobDetailsSection: document.getElementById('jobDetailsSection'),
    jobForm: document.getElementById('jobForm'),
    backBtn: document.getElementById('backBtn'),
    targetJob: document.getElementById('targetJob'),
    targetLocation: document.getElementById('targetLocation'),
    industry: document.getElementById('industry'),
    experienceLevel: document.getElementById('experienceLevel'),
    
    // Results Section
    resultsSection: document.getElementById('resultsSection'),
    resultsLoading: document.getElementById('resultsLoading'),
    resultsContent: document.getElementById('resultsContent'),
    resultsActions: document.getElementById('resultsActions'),
    startOverBtn: document.getElementById('startOverBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    
    // Progress Steps
    progressSteps: document.querySelectorAll('.progress-step')
};

// API Configuration
const API_URL =  'https://viqri-cv-api-5u7hdc64va-uc.a.run.app';

// ============================================
// Initialization
// ============================================
function init() {
    if (state.initialized) {
        console.log('⚠️ Already initialized, skipping...');
        return;
    }
    
    console.log('🚀 Starting initialization...');
    setupEventListeners();
    updateProgressIndicator(1);
    state.initialized = true;
    console.log('✅ Viqri CV - AI Template Generator Ready!');
    console.log('🔗 Backend API:', API_URL);
    console.log('🤖 Powered by Groq Llama 3.1');
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    console.log('📡 Setting up event listeners...');
    
    // Upload zone click
    if (elements.uploadZone) {
        console.log('✅ Setting up uploadZone click listener');
        elements.uploadZone.addEventListener('click', (e) => {
            console.log('🖱️ Upload zone clicked!', e.target);
            // Don't trigger file input if clicking on change file button
            if (e.target.closest('#changeFile')) {
                console.log('   -> Clicked on change file button, ignoring');
                return;
            }
            if (elements.fileInput) {
                console.log('   -> Triggering file input click');
                elements.fileInput.click();
            } else {
                console.error('   -> fileInput not found!');
            }
        });
        console.log('✅ Upload zone click listener added');
    } else {
        console.error('❌ uploadZone element not found!');
    }
    
    // File input change
    if (elements.fileInput) {
        console.log('✅ Setting up fileInput change listener');
        elements.fileInput.addEventListener('change', handleFileSelect);
    } else {
        console.error('❌ fileInput element not found!');
    }
    
    // Drag and drop
    if (elements.uploadZone) {
        console.log('✅ Setting up drag and drop listeners');
        elements.uploadZone.addEventListener('dragover', handleDragOver);
        elements.uploadZone.addEventListener('dragleave', handleDragLeave);
        elements.uploadZone.addEventListener('drop', handleDrop);
    }
    
    // Change file button
    if (elements.changeFile) {
        console.log('✅ Setting up changeFile listener');
        elements.changeFile.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUpload();
        });
    }
    
    // Continue button
    if (elements.continueBtn) {
        console.log('✅ Setting up continueBtn listener');
        elements.continueBtn.addEventListener('click', () => {
            goToStep(2);
        });
    }
    
    // Back button
    if (elements.backBtn) {
        console.log('✅ Setting up backBtn listener');
        elements.backBtn.addEventListener('click', () => {
            goToStep(1);
        });
    }
    
    // Job form submit
    if (elements.jobForm) {
        console.log('✅ Setting up jobForm submit listener');
        elements.jobForm.addEventListener('submit', handleJobFormSubmit);
    }
    
    // Start over button
    if (elements.startOverBtn) {
        console.log('✅ Setting up startOverBtn listener');
        elements.startOverBtn.addEventListener('click', startOver);
    }
    
    // Download button
    if (elements.downloadBtn) {
        console.log('✅ Setting up downloadBtn listener');
        elements.downloadBtn.addEventListener('click', downloadReport);
    }
    
    // Prevent default drag behavior globally
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });
    
    console.log('✅ All event listeners set up successfully!');
}

// ============================================
// File Upload Handlers
// ============================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndProcessFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    if (elements.uploadZone) {
        elements.uploadZone.style.borderColor = 'rgba(99, 102, 241, 0.6)';
        elements.uploadZone.style.transform = 'translateY(-2px) scale(1.01)';
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    if (elements.uploadZone) {
        elements.uploadZone.style.borderColor = '';
        elements.uploadZone.style.transform = '';
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    if (elements.uploadZone) {
        elements.uploadZone.style.borderColor = '';
        elements.uploadZone.style.transform = '';
    }
    
    const file = e.dataTransfer.files[0];
    if (file) {
        validateAndProcessFile(file);
    }
}

function validateAndProcessFile(file) {
    // Validate file type
    const validTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a PDF, DOC, or DOCX file.', 'error');
        return;
    }
    
    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
        showNotification('File size must be less than 10MB.', 'error');
        return;
    }
    
    state.uploadedFile = file;
    processFile(file);
}

function processFile(file) {
    // Show processing state
    if (elements.uploadIdle) elements.uploadIdle.style.display = 'none';
    if (elements.uploadSuccess) elements.uploadSuccess.style.display = 'none';
    if (elements.uploadProcessing) elements.uploadProcessing.style.display = 'block';
    if (elements.continueBtn) elements.continueBtn.style.display = 'none';
    
    // Simulate processing with progress animation
    let progress = 0;
    const processingSteps = [
        'Reading document structure...',
        'Extracting text content...',
        'Analyzing CV format...',
        'Identifying key sections...',
        'Processing complete!'
    ];
    
    let stepIndex = 0;
    const processingStepElement = elements.uploadProcessing?.querySelector('.processing-step');
    
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 90) {
            progress = 90;
        }
        updateProgressBar(progress);
        
        // Update processing step text
        if (processingStepElement && stepIndex < processingSteps.length) {
            const stepProgress = Math.floor((progress / 90) * processingSteps.length);
            if (stepProgress > stepIndex) {
                stepIndex = stepProgress;
                processingStepElement.textContent = processingSteps[Math.min(stepIndex, processingSteps.length - 1)];
            }
        }
    }, 300);
    
    // Upload to API
    uploadToAPI(file, progressInterval);
}

async function uploadToAPI(file, progressInterval) {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        updateProgressBar(100);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
        }

        const result = await response.json();

        if (result.success) {
            state.cvData = result.data;
            console.log('✅ CV Parsed Successfully!', state.cvData);
            
            setTimeout(() => {
                showUploadSuccess(file.name);
            }, 500);
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('❌ Upload error:', error);
        showNotification('Failed to analyze CV: ' + error.message, 'error');
        clearInterval(progressInterval);
        resetUpload();
    }
}

function updateProgressBar(progress) {
    const percent = Math.min(Math.round(progress), 100);
    if (elements.progressBar) {
        elements.progressBar.style.width = `${percent}%`;
    }
}

function showUploadSuccess(filename) {
    if (elements.uploadProcessing) elements.uploadProcessing.style.display = 'none';
    if (elements.uploadSuccess) elements.uploadSuccess.style.display = 'block';
    if (elements.successFilename) elements.successFilename.textContent = filename;
    if (elements.continueBtn) {
        elements.continueBtn.style.display = 'inline-flex';
    }
    
    showNotification('CV analyzed successfully!', 'success');
}

function resetUpload() {
    state.uploadedFile = null;
    state.cvData = null;
    
    if (elements.fileInput) elements.fileInput.value = '';
    if (elements.uploadIdle) elements.uploadIdle.style.display = 'block';
    if (elements.uploadProcessing) elements.uploadProcessing.style.display = 'none';
    if (elements.uploadSuccess) elements.uploadSuccess.style.display = 'none';
    if (elements.continueBtn) elements.continueBtn.style.display = 'none';
    if (elements.progressBar) elements.progressBar.style.width = '0%';
}

// ============================================
// Navigation
// ============================================
function goToStep(step) {
    // Hide all sections
    if (elements.uploadSection) elements.uploadSection.style.display = 'none';
    if (elements.jobDetailsSection) elements.jobDetailsSection.style.display = 'none';
    if (elements.resultsSection) elements.resultsSection.style.display = 'none';
    
    // Remove active class from all sections
    document.querySelectorAll('.step-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show requested section
    let activeSection = null;
    if (step === 1) {
        if (elements.uploadSection) {
            elements.uploadSection.style.display = 'block';
            elements.uploadSection.classList.add('active');
            activeSection = elements.uploadSection;
        }
    } else if (step === 2) {
        if (elements.jobDetailsSection) {
            elements.jobDetailsSection.style.display = 'block';
            elements.jobDetailsSection.classList.add('active');
            activeSection = elements.jobDetailsSection;
        }
    } else if (step === 3) {
        if (elements.resultsSection) {
            elements.resultsSection.style.display = 'block';
            elements.resultsSection.classList.add('active');
            activeSection = elements.resultsSection;
        }
    }
    
    // Update progress indicator
    updateProgressIndicator(step);
    
    // Scroll to section
    if (activeSection) {
        setTimeout(() => {
            activeSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
    
    state.currentStep = step;
}

function updateProgressIndicator(step) {
    elements.progressSteps.forEach((stepElement, index) => {
        const stepNumber = index + 1;
        if (stepNumber <= step) {
            stepElement.classList.add('active');
        } else {
            stepElement.classList.remove('active');
        }
    });
}

// ============================================
// Job Form Handler
// ============================================
async function handleJobFormSubmit(e) {
    e.preventDefault();

    const targetJob = elements.targetJob?.value.trim();
    const targetLocation = elements.targetLocation?.value.trim();
    const industry = elements.industry?.value;
    const experienceLevel = elements.experienceLevel?.value;

    if (!targetJob || !targetLocation) {
        showNotification('Please fill in required fields', 'error');
        return;
    }

    // Save job details
    state.jobDetails = {
        target_job: targetJob,
        target_location: targetLocation,
        industry: industry || null,
        experience_level: experienceLevel || null
    };

    // Show results section with loading
    goToStep(3);
    
    if (elements.resultsLoading) elements.resultsLoading.style.display = 'block';
    if (elements.resultsContent) elements.resultsContent.style.display = 'none';
    if (elements.resultsActions) elements.resultsActions.style.display = 'none';

    // Animate loading steps
    const loadingSteps = [
        'Processing career trajectory...',
        'Analyzing market trends...',
        'Generating recommendations...',
        'Finalizing strategy...'
    ];
    
    let stepIndex = 0;
    const loadingStepElement = elements.resultsLoading?.querySelector('.loading-step');
    
    const stepInterval = setInterval(() => {
        if (loadingStepElement && stepIndex < loadingSteps.length) {
            loadingStepElement.textContent = loadingSteps[stepIndex];
            stepIndex++;
        }
    }, 1500);

    // Generate template
    try {
        const response = await fetch(`${API_URL}/api/generate-template`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cv_data: state.cvData,
                target_job: targetJob,
                target_location: targetLocation,
                industry: industry || null,
                experience_level: experienceLevel || null
            })
        });

        clearInterval(stepInterval);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Template generation failed');
        }

        const result = await response.json();

        if (result.success) {
            console.log('✅ Template Generated!', result.data);
            
            // Save data to localStorage
            localStorage.setItem('templateData', JSON.stringify(result.data));
            localStorage.setItem('cvData', JSON.stringify(state.cvData));
            localStorage.setItem('jobDetails', JSON.stringify(state.jobDetails));
            
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
        clearInterval(stepInterval);
        if (elements.resultsLoading) elements.resultsLoading.style.display = 'none';
        if (elements.resultsActions) elements.resultsActions.style.display = 'flex';
    }
}

// ============================================
// Results Actions
// ============================================
function startOver() {
    resetUpload();
    
    if (elements.jobForm) elements.jobForm.reset();
    
    state.cvData = null;
    state.jobDetails = null;
    
    goToStep(1);
}

function downloadReport() {
    showNotification('Download feature coming soon!', 'info');
}

// ============================================
// Notifications
// ============================================
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Add icon based on type
    let icon = '';
    if (type === 'success') {
        icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (type === 'error') {
        icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    } else {
        icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            ${icon}
            <span>${message}</span>
        </div>
    `;
    
    Object.assign(notification.style, {
        position: 'fixed',
        top: '2rem',
        right: '2rem',
        padding: '1rem 1.5rem',
        borderRadius: '16px',
        background: type === 'success' ? 'linear-gradient(135deg, #10b981, #059669)' : 
                   type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 
                   'linear-gradient(135deg, #3b82f6, #2563eb)',
        color: 'white',
        fontWeight: '600',
        fontSize: '0.95rem',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.2)',
        zIndex: '10000',
        animation: 'slideInRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        maxWidth: '400px',
        backdropFilter: 'blur(10px)'
    });

    // Add styles if not already present
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideInRight {
                from { 
                    transform: translateX(400px); 
                    opacity: 0; 
                }
                to { 
                    transform: translateX(0); 
                    opacity: 1; 
                }
            }
            @keyframes slideOutRight {
                from { 
                    transform: translateX(0); 
                    opacity: 1; 
                }
                to { 
                    transform: translateX(400px); 
                    opacity: 0; 
                }
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

// ============================================
// Page Load Animation
// ============================================
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease-in';
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// ============================================
// Initialize on DOM Ready - BULLETPROOF VERSION
// ============================================
function safeInit() {
    console.log('🔄 Initializing Viqri CV...');
    console.log('Document ready state:', document.readyState);
    
    // Double-check all critical elements exist
    const criticalElements = {
        uploadZone: document.getElementById('uploadZone'),
        fileInput: document.getElementById('fileInput'),
        uploadIdle: document.getElementById('uploadIdle')
    };
    
    console.log('Critical elements check:', criticalElements);
    
    if (!criticalElements.uploadZone) {
        console.error('❌ CRITICAL: uploadZone not found!');
        console.log('Available elements:', document.body.innerHTML.substring(0, 500));
        return;
    }
    
    if (!criticalElements.fileInput) {
        console.error('❌ CRITICAL: fileInput not found!');
        return;
    }
    
    console.log('✅ All critical elements found, proceeding with initialization...');
    init();
}

// Try multiple methods to ensure initialization
if (document.readyState === 'loading') {
    console.log('Document still loading, adding DOMContentLoaded listener...');
    document.addEventListener('DOMContentLoaded', safeInit);
} else if (document.readyState === 'interactive' || document.readyState === 'complete') {
    console.log('Document already loaded, initializing immediately...');
    safeInit();
}

// Backup: try again after a short delay
setTimeout(() => {
    if (!state.initialized) {
        console.log('⚠️ Backup initialization attempt...');
        safeInit();
    }
}, 500);