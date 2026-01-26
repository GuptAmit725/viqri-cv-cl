// Get data from localStorage or URL params
let templateData = null;
let cvData = null;
let jobDetails = null;

// Initialize
init();

function init() {
    loadData();
    setupEventListeners();
    setupSmoothScroll();
}

function loadData() {
    try {
        // Try to get data from localStorage
        const storedTemplateData = localStorage.getItem('templateData');
        const storedCvData = localStorage.getItem('cvData');
        const storedJobDetails = localStorage.getItem('jobDetails');

        if (storedTemplateData) {
            templateData = JSON.parse(storedTemplateData);
            cvData = JSON.parse(storedCvData);
            jobDetails = JSON.parse(storedJobDetails);
            
            renderTemplate();
        } else {
            showError('No template data found. Please generate a template first.');
        }
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Error loading template data.');
    }
}

function setupEventListeners() {
    // Back button
    document.getElementById('backBtn').addEventListener('click', () => {
        window.location.href = 'index.html';
    });

    // Edit button
    document.getElementById('editBtn').addEventListener('click', () => {
        window.location.href = 'edit-template.html';
    });

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', downloadPDF);

    // Print button
    document.getElementById('printBtn').addEventListener('click', () => {
        window.print();
    });

    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Update active link
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
}

function setupSmoothScroll() {
    // Intersection Observer for nav highlighting
    const sections = document.querySelectorAll('.content-section');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, { threshold: 0.5 });

    sections.forEach(section => observer.observe(section));
}

function renderTemplate() {
    // Hide loading, show content
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('contentSections').style.display = 'block';

    // Check if template data is empty or incomplete
    const hasTemplateData = templateData && 
                           templateData.template_structure && 
                           Object.keys(templateData.template_structure).length > 0;
    
    if (!hasTemplateData) {
        // Show edit prompt at top
        const overview = document.getElementById('overview');
        const editPrompt = document.createElement('div');
        editPrompt.style.cssText = `
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
        `;
        editPrompt.innerHTML = `
            <h3 style="margin-bottom: 1rem; font-size: 1.5rem;">📝 Your CV is Ready to Edit!</h3>
            <p style="margin-bottom: 1.5rem; opacity: 0.9;">
                AI recommendations are being generated. Start editing your CV now to customize it for your target role.
            </p>
            <button onclick="window.location.href='edit-template.html'" 
                    style="background: white; color: #667eea; padding: 1rem 2rem; border: none; border-radius: 10px; font-weight: 600; cursor: pointer; font-size: 1rem;">
                ✏️ Edit Your CV Now
            </button>
        `;
        overview.insertBefore(editPrompt, overview.firstChild);
    }

    // Render job details
    renderJobDetails();

    // Render statistics
    renderStatistics();

    // Render template structure
    renderTemplateStructure();

    // Render content recommendations
    renderContentRecommendations();

    // Render location insights
    renderLocationInsights();

    // Render industry insights
    renderIndustryInsights();

    // Render action items
    renderActionItems();

    // Render CV preview
    renderCVPreview();

    console.log('✅ Template rendered successfully!');
}

function renderJobDetails() {
    if (!jobDetails) return;

    document.getElementById('targetJobDisplay').textContent = jobDetails.target_job || 'Not specified';
    document.getElementById('targetLocationDisplay').textContent = jobDetails.target_location || 'Not specified';
    document.getElementById('industryDisplay').textContent = jobDetails.industry || 'Not specified';
    
    const experienceLevels = {
        'entry': 'Entry Level',
        'mid': 'Mid Level',
        'senior': 'Senior Level',
        'lead': 'Lead/Principal',
        'executive': 'Executive'
    };
    document.getElementById('experienceDisplay').textContent = 
        experienceLevels[jobDetails.experience_level] || 'Not specified';
}

function renderStatistics() {
    const rec = templateData.content_recommendations || {};
    
    // If template data is empty, calculate from CV data directly
    if (!rec.key_skills || rec.key_skills.length === 0) {
        // Count skills from actual CV
        const skills = cvData.skills || {};
        const allSkills = [
            ...(skills.programming_languages || []),
            ...(skills.frameworks || []),
            ...(skills.tools || []),
            ...(skills.databases || []),
            ...(skills.cloud || []),
            ...(skills.technical || [])
        ];
        document.getElementById('keySkillsCount').textContent = allSkills.length;
        document.getElementById('missingSkillsCount').textContent = 0;
        document.getElementById('actionItemsCount').textContent = 'N/A';
        document.getElementById('keywordsCount').textContent = 0;
    } else {
        // Use template recommendations if available
        document.getElementById('keySkillsCount').textContent = rec.key_skills?.length || 0;
        document.getElementById('missingSkillsCount').textContent = rec.missing_skills?.length || 0;
        
        const actions = templateData.action_items || {};
        const totalActions = (actions.immediate?.length || 0) + 
                            (actions.important?.length || 0) + 
                            (actions.nice_to_have?.length || 0);
        document.getElementById('actionItemsCount').textContent = totalActions;
        document.getElementById('keywordsCount').textContent = rec.keywords?.length || 0;
    }
}

function renderTemplateStructure() {
    const structure = templateData.template_structure;
    
    // If no template data, show message
    if (!structure || Object.keys(structure).length === 0) {
        document.getElementById('structureContent').innerHTML = `
            <div class="info-card">
                <p style="color: var(--color-text-light);">
                    AI recommendations are being generated. Meanwhile, you can edit your CV using the "Edit Template" button above.
                </p>
            </div>
        `;
        return;
    }

    let html = '<div class="info-card">';
    
    if (structure.format) {
        html += `<h3>📋 Recommended Format</h3>`;
        html += `<p><strong>Format Type:</strong> ${structure.format}</p>`;
    }
    
    if (structure.length) {
        html += `<p><strong>Optimal Length:</strong> ${structure.length}</p>`;
    }
    
    if (structure.sections && structure.sections.length > 0) {
        html += `<h3 style="margin-top: 1.5rem;">📑 Section Order</h3>`;
        html += '<ol style="margin-left: 1.5rem; margin-top: 1rem;">';
        structure.sections.forEach(section => {
            html += `<li style="margin-bottom: 0.5rem;">${section}</li>`;
        });
        html += '</ol>';
    }
    
    if (structure.section_priorities) {
        html += `<h3 style="margin-top: 1.5rem;">⭐ Section Priorities</h3>`;
        html += '<ul style="margin-left: 1.5rem; margin-top: 1rem;">';
        Object.entries(structure.section_priorities).forEach(([section, reason]) => {
            html += `<li style="margin-bottom: 0.75rem;"><strong>${section}:</strong> ${reason}</li>`;
        });
        html += '</ul>';
    }
    
    html += '</div>';
    
    document.getElementById('structureContent').innerHTML = html;
}

function renderContentRecommendations() {
    const rec = templateData.content_recommendations;
    if (!rec) return;

    let html = '';

    // Professional Summary
    if (rec.summary) {
        html += `
            <div class="info-card">
                <h3>✍️ Professional Summary</h3>
                <p style="font-size: 1.05rem; line-height: 1.7; font-style: italic; color: var(--color-text);">
                    "${rec.summary}"
                </p>
            </div>
        `;
    }

    // Key Skills
    if (rec.key_skills && rec.key_skills.length > 0) {
        html += `
            <div class="info-card">
                <h3>🎯 Key Skills to Highlight</h3>
                <p>These skills should be prominently featured in your CV:</p>
                <div class="skills-grid">
                    ${rec.key_skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
        `;
    }

    // Missing Skills
    if (rec.missing_skills && rec.missing_skills.length > 0) {
        html += `
            <div class="warning-box">
                <h4>⚠️ Skills to Add or Develop</h4>
                <ul>
                    ${rec.missing_skills.map(skill => `<li>${skill}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Experience Tips
    if (rec.experience_tips && rec.experience_tips.length > 0) {
        html += `
            <div class="info-card">
                <h3>💼 Experience Section Tips</h3>
                <ul>
                    ${rec.experience_tips.map(tip => `<li>${tip}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // ATS Keywords
    if (rec.keywords && rec.keywords.length > 0) {
        html += `
            <div class="info-card">
                <h3>🔑 ATS Keywords</h3>
                <p>Include these keywords to pass Applicant Tracking Systems:</p>
                <div class="skills-grid">
                    ${rec.keywords.map(keyword => `<span class="skill-tag skill-tag-secondary">${keyword}</span>`).join('')}
                </div>
            </div>
        `;
    }

    document.getElementById('contentRecommendations').innerHTML = html;
}

function renderLocationInsights() {
    const location = templateData.location_specific;
    if (!location) return;

    let html = '';

    if (location.format_preferences) {
        html += `
            <div class="info-card">
                <h3>📄 Regional Format Preferences</h3>
                <p>${location.format_preferences}</p>
            </div>
        `;
    }

    if (location.cultural_considerations && location.cultural_considerations.length > 0) {
        html += `
            <div class="info-card">
                <h3>🌏 Cultural Considerations</h3>
                <ul>
                    ${location.cultural_considerations.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (location.common_requirements && location.common_requirements.length > 0) {
        html += `
            <div class="info-card">
                <h3>✅ Common Requirements</h3>
                <p>What employers in ${jobDetails.target_location} typically expect:</p>
                <ul>
                    ${location.common_requirements.map(req => `<li>${req}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    document.getElementById('locationContent').innerHTML = html || '<p>No location-specific insights available.</p>';
}

function renderIndustryInsights() {
    const industry = templateData.industry_insights;
    if (!industry) return;

    let html = '';

    if (industry.trends) {
        html += `
            <div class="info-card">
                <h3>📈 Industry Trends</h3>
                <p>${industry.trends}</p>
            </div>
        `;
    }

    if (industry.sought_after_skills && industry.sought_after_skills.length > 0) {
        html += `
            <div class="info-card">
                <h3>🔥 Most Sought-After Skills</h3>
                <div class="skills-grid">
                    ${industry.sought_after_skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
        `;
    }

    if (industry.red_flags && industry.red_flags.length > 0) {
        html += `
            <div class="warning-box">
                <h4>🚫 Things to Avoid</h4>
                <ul>
                    ${industry.red_flags.map(flag => `<li>${flag}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    document.getElementById('industryContent').innerHTML = html || '<p>No industry insights available.</p>';
}

function renderActionItems() {
    const actions = templateData.action_items;
    if (!actions) return;

    let html = '';

    if (actions.immediate && actions.immediate.length > 0) {
        html += `
            <div class="action-priority priority-high">
                <h3>🔥 Do This Now</h3>
                <ul>
                    ${actions.immediate.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (actions.important && actions.important.length > 0) {
        html += `
            <div class="action-priority priority-medium">
                <h3>⚡ Important Changes</h3>
                <ul>
                    ${actions.important.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    if (actions.nice_to_have && actions.nice_to_have.length > 0) {
        html += `
            <div class="action-priority priority-low">
                <h3>✨ Nice to Have</h3>
                <ul>
                    ${actions.nice_to_have.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    document.getElementById('actionItemsContent').innerHTML = html;
}

function renderCVPreview() {
    if (!cvData || !templateData) return;

    const personal = cvData.personal_info || {};
    const rec = templateData.content_recommendations || {};
    
    let html = `
        <div class="cv-header">
            <h1 class="cv-name">${personal.name || '[Your Name]'}</h1>
            <p class="cv-title">${jobDetails.target_job || '[Target Job Title]'}</p>
            <div class="cv-contact">
                ${personal.email ? `<span>📧 ${personal.email}</span>` : ''}
                ${personal.phone ? `<span>📱 ${personal.phone}</span>` : ''}
                ${personal.location ? `<span>📍 ${personal.location}</span>` : ''}
                ${personal.linkedin ? `<span>💼 ${personal.linkedin}</span>` : ''}
            </div>
        </div>

        ${rec.summary ? `
            <div class="cv-section">
                <h2 class="cv-section-title">Professional Summary</h2>
                <p>${rec.summary}</p>
            </div>
        ` : ''}

        ${rec.key_skills && rec.key_skills.length > 0 ? `
            <div class="cv-section">
                <h2 class="cv-section-title">Key Skills</h2>
                <div class="skills-grid">
                    ${rec.key_skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                </div>
            </div>
        ` : ''}

        ${cvData.experience && cvData.experience.length > 0 ? `
            <div class="cv-section">
                <h2 class="cv-section-title">Experience</h2>
                ${cvData.experience.slice(0, 2).map(exp => `
                    <div style="margin-bottom: 1.5rem;">
                        <h3 style="color: var(--color-primary); margin-bottom: 0.25rem;">
                            ${exp.title || '[Job Title]'}
                        </h3>
                        <p style="color: var(--color-accent); margin-bottom: 0.5rem;">
                            ${exp.company || '[Company]'} ${exp.duration ? '• ' + exp.duration : ''}
                        </p>
                        ${exp.responsibilities && exp.responsibilities.length > 0 ? `
                            <ul style="margin-left: 1.5rem;">
                                ${exp.responsibilities.slice(0, 3).map(resp => `<li>${resp}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        ` : ''}

        ${cvData.education && cvData.education.length > 0 ? `
            <div class="cv-section">
                <h2 class="cv-section-title">Education</h2>
                ${cvData.education.map(edu => `
                    <div style="margin-bottom: 1rem;">
                        <h3 style="color: var(--color-primary); font-size: 1rem; margin-bottom: 0.25rem;">
                            ${edu.degree || edu.institution || '[Degree]'}
                        </h3>
                        <p style="color: var(--color-text-light); font-size: 0.9rem;">
                            ${edu.institution || ''} ${edu.year ? '• ' + edu.year : ''}
                        </p>
                    </div>
                `).join('')}
            </div>
        ` : ''}
    `;

    document.getElementById('cvPreviewContent').innerHTML = html;
}

function downloadPDF() {
    // For now, trigger print dialog
    // In production, you'd use a library like jsPDF or html2pdf
    window.print();
    
    // Alternative: Show message
    setTimeout(() => {
        alert('Use the Print dialog to save as PDF, or implement a PDF generation library for direct download.');
    }, 500);
}

function showError(message) {
    const loadingState = document.getElementById('loadingState');
    loadingState.innerHTML = `
        <div style="text-align: center;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#f56565" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <p style="margin-top: 1.5rem; color: #f56565; font-weight: 600;">${message}</p>
            <button onclick="window.location.href='index.html'" 
                    style="margin-top: 1rem; padding: 0.75rem 1.5rem; background: #e94560; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;">
                Go Back
            </button>
        </div>
    `;
}

console.log('📄 Template Page Ready!');
console.log('Template Data:', templateData);
console.log('CV Data:', cvData);
console.log('Job Details:', jobDetails);

// ============================================================================
// JOB MATCHING FUNCTIONALITY
// ============================================================================

const API_BASE_URL = 'http://localhost:8000';
let currentJobs = [];
let currentInsights = null;

function initJobMatching() {
    console.log('🎯 Initializing job matching...');
    
    const searchJobsBtn = document.getElementById('searchJobsBtn');
    const refreshJobsBtn = document.getElementById('refreshJobsBtn');
    const retryJobsBtn = document.getElementById('retryJobsBtn');
    
    if (searchJobsBtn) searchJobsBtn.addEventListener('click', searchForJobs);
    if (refreshJobsBtn) refreshJobsBtn.addEventListener('click', searchForJobs);
    if (retryJobsBtn) retryJobsBtn.addEventListener('click', searchForJobs);
    
    console.log('✅ Job matching initialized');
}

async function searchForJobs() {
    console.log('🔍 Searching for jobs...');
    
    const targetJob = jobDetails?.target_job || localStorage.getItem('targetJob') || '';
    const targetLocation = jobDetails?.target_location || localStorage.getItem('targetLocation') || '';
    const experienceLevel = jobDetails?.experience_level || localStorage.getItem('experienceLevel') || '';
    
    if (!cvData || Object.keys(cvData).length === 0) {
        showJobError('No CV data found. Please upload your CV first.');
        return;
    }
    
    if (!targetJob || !targetLocation) {
        showJobError('Job title and location are required. Please go back and fill in the details.');
        return;
    }
    
    showJobLoading();
    
    try {
        console.log('📡 Calling job matching API...');
        const response = await fetch(`${API_BASE_URL}/api/match-jobs`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                cv_data: cvData,
                job_title: targetJob,
                location: targetLocation,
                experience_level: experienceLevel
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch jobs');
        }
        
        const data = await response.json();
        console.log('✅ Jobs received:', data);
        
        if (data.success && data.jobs && data.jobs.length > 0) {
            currentJobs = data.jobs;
            currentInsights = data.insights;
            displayJobs(data.jobs, data.insights);
        } else {
            showJobError(data.message || 'No jobs found matching your criteria.');
        }
        
    } catch (error) {
        console.error('❌ Job search error:', error);
        showJobError(error.message || 'Failed to search for jobs. Please try again.');
    }
}

function showJobLoading() {
    document.getElementById('jobInitialState').style.display = 'none';
    document.getElementById('jobErrorState').style.display = 'none';
    document.getElementById('jobsContainer').style.display = 'none';
    document.getElementById('jobLoadingState').style.display = 'block';
}

function showJobError(message) {
    document.getElementById('jobInitialState').style.display = 'none';
    document.getElementById('jobLoadingState').style.display = 'none';
    document.getElementById('jobsContainer').style.display = 'none';
    document.getElementById('jobErrorState').style.display = 'block';
    document.getElementById('jobErrorMessage').textContent = message;
}

function displayJobs(jobs, insights) {
    console.log('📋 Displaying jobs:', jobs.length);
    
    document.getElementById('jobInitialState').style.display = 'none';
    document.getElementById('jobLoadingState').style.display = 'none';
    document.getElementById('jobErrorState').style.display = 'none';
    document.getElementById('jobsContainer').style.display = 'block';
    
    document.getElementById('jobsFoundText').textContent = `Found ${jobs.length} matching opportunities`;
    
    const jobsGrid = document.getElementById('jobsGrid');
    jobsGrid.innerHTML = '';
    
    jobs.forEach((job, index) => {
        const jobCard = createJobCard(job, index + 1);
        jobsGrid.appendChild(jobCard);
    });
    
    if (insights) displayInsights(insights);
    
    document.getElementById('matching-jobs').scrollIntoView({ behavior: 'smooth' });
}

function createJobCard(job, rank) {
    const card = document.createElement('div');
    card.className = 'job-card';
    
    const score = job.match_score || 0;
    let scoreClass = 'match-low';
    if (score >= 80) scoreClass = 'match-high';
    else if (score >= 60) scoreClass = 'match-medium';
    
    card.innerHTML = `
        <div class="job-card-header">
            <div class="job-rank">#${rank}</div>
            <div class="job-match-score ${scoreClass}">
                <span class="score-value">${score}%</span>
                <span class="score-label">Match</span>
            </div>
        </div>
        
        <h3 class="job-title">${job.title}</h3>
        <div class="job-company">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
            </svg>
            ${job.company}
        </div>
        <div class="job-location">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
            </svg>
            ${job.location}
        </div>
        
        ${job.summary ? `<p class="job-summary">${job.summary}</p>` : ''}
        
        ${job.fit_reason ? `
        <div class="job-fit-reason">
            <strong>Why it's a good fit:</strong> ${job.fit_reason}
        </div>
        ` : ''}
        
        ${job.skills_to_highlight && job.skills_to_highlight.length > 0 ? `
        <div class="job-skills-highlight">
            <strong>Highlight these skills:</strong>
            <div class="skills-tags">
                ${job.skills_to_highlight.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
            </div>
        </div>
        ` : ''}
        
        <div class="job-card-footer">
            <span class="job-posted">${job.posted_date || 'Recently'}</span>
            <a href="${job.url}" target="_blank" rel="noopener noreferrer" class="btn-apply">
                Apply on LinkedIn
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                </svg>
            </a>
        </div>
    `;
    
    return card;
}

function displayInsights(insights) {
    const insightsContent = document.getElementById('insightsContent');
    
    insightsContent.innerHTML = `
        <div class="insight-item">
            <h4>📊 Match Quality</h4>
            <p class="insight-highlight">${insights.match_quality || 'Good'}</p>
            <p>Average match score: <strong>${insights.average_match_score || 65}%</strong></p>
        </div>
        
        ${insights.recommendations && insights.recommendations.length > 0 ? `
        <div class="insight-item">
            <h4>💡 Recommendations</h4>
            <ul class="insight-list">
                ${insights.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${insights.action_items && insights.action_items.length > 0 ? `
        <div class="insight-item">
            <h4>✅ Action Items</h4>
            <ul class="insight-list action-list">
                ${insights.action_items.map(action => `<li>${action}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        ${insights.skills_in_demand && insights.skills_in_demand.length > 0 ? `
        <div class="insight-item">
            <h4>🔥 Skills in Demand</h4>
            <div class="skills-tags">
                ${insights.skills_in_demand.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
            </div>
        </div>
        ` : ''}
    `;
}

initJobMatching();