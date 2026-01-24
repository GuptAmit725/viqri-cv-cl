// Global state
let cvData = null;
let templateData = null;
let jobDetails = null;

// Initialize
init();

function init() {
    loadData();
    setupEventListeners();
    populateFields();
}

function loadData() {
    try {
        cvData = JSON.parse(localStorage.getItem('cvData') || '{}');
        templateData = JSON.parse(localStorage.getItem('templateData') || '{}');
        jobDetails = JSON.parse(localStorage.getItem('jobDetails') || '{}');
    } catch (error) {
        console.error('Error loading data:', error);
        showNotification('Error loading CV data', 'error');
    }
}

function setupEventListeners() {
    // Header buttons
    document.getElementById('backBtn').addEventListener('click', () => {
        window.location.href = 'template.html';
    });
    
    document.getElementById('viewModeBtn').addEventListener('click', () => {
        saveChanges();
        window.location.href = 'template.html';
    });
    
    document.getElementById('saveBtn').addEventListener('click', saveChanges);
    document.getElementById('generateFinalBtn').addEventListener('click', generateFinalCV);
    
    // Add buttons
    document.getElementById('addExperienceBtn').addEventListener('click', () => addExperienceItem());
    document.getElementById('addEducationBtn').addEventListener('click', () => addEducationItem());
    document.getElementById('addProjectBtn').addEventListener('click', () => addProjectItem());
    document.getElementById('addCertificationBtn').addEventListener('click', () => addCertificationItem());
    document.getElementById('addAwardBtn').addEventListener('click', () => addAwardItem());
    
    // Tag inputs
    document.querySelectorAll('.tag-input').forEach(input => {
        input.addEventListener('keypress', handleTagInput);
    });
}

function populateFields() {
    if (!cvData) return;
    
    // Personal Info
    const personal = cvData.personal_info || {};
    document.getElementById('name').value = personal.name || '';
    document.getElementById('email').value = personal.email || '';
    document.getElementById('phone').value = personal.phone || '';
    document.getElementById('location').value = personal.location || '';
    document.getElementById('linkedin').value = personal.linkedin || '';
    document.getElementById('github').value = personal.github || '';
    document.getElementById('website').value = personal.website || '';
    
    // Summary
    document.getElementById('summary').value = cvData.professional_summary || '';
    
    // Experience
    const experience = cvData.experience || [];
    experience.forEach(exp => addExperienceItem(exp));
    
    // Education
    const education = cvData.education || [];
    education.forEach(edu => addEducationItem(edu));
    
    // Skills
    const skills = cvData.skills || {};
    populateTags('programmingLanguages', skills.programming_languages || []);
    populateTags('frameworks', skills.frameworks || []);
    populateTags('tools', skills.tools || []);
    populateTags('databases', skills.databases || []);
    populateTags('cloud', skills.cloud || []);
    
    // Projects
    const projects = cvData.projects || [];
    projects.forEach(proj => addProjectItem(proj));
    
    // Certifications
    const certifications = cvData.certifications || [];
    certifications.forEach(cert => addCertificationItem(cert));
    
    // Awards
    const awards = cvData.awards || [];
    awards.forEach(award => addAwardItem(award));
}

// Experience Functions
function addExperienceItem(data = null) {
    const id = 'exp_' + Date.now();
    const container = document.getElementById('experienceList');
    
    const card = document.createElement('div');
    card.className = 'item-card';
    card.id = id;
    
    card.innerHTML = `
        <div class="item-header" onclick="toggleCard('${id}')">
            <div class="item-title">${data?.title || 'New Experience'}</div>
            <div class="item-actions">
                <button class="btn-icon" onclick="event.stopPropagation(); toggleCard('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="18 15 12 9 6 15"/>
                    </svg>
                </button>
                <button class="btn-icon danger" onclick="event.stopPropagation(); removeItem('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        </div>
        <div class="item-fields">
            <div class="field-row">
                <div class="field-group">
                    <label>Job Title *</label>
                    <input type="text" class="exp-title" value="${data?.title || ''}" placeholder="Senior Software Engineer" required>
                </div>
                <div class="field-group">
                    <label>Company *</label>
                    <input type="text" class="exp-company" value="${data?.company || ''}" placeholder="Company Name" required>
                </div>
            </div>
            <div class="field-row">
                <div class="field-group">
                    <label>Start Date</label>
                    <input type="text" class="exp-start" value="${data?.start_date || ''}" placeholder="Jan 2020">
                </div>
                <div class="field-group">
                    <label>End Date</label>
                    <input type="text" class="exp-end" value="${data?.end_date || ''}" placeholder="Present">
                </div>
            </div>
            <div class="field-group">
                <label>Location</label>
                <input type="text" class="exp-location" value="${data?.location || ''}" placeholder="City, Country">
            </div>
            <div class="field-group">
                <label>Responsibilities & Achievements</label>
                <div class="responsibilities-list" id="${id}_responsibilities">
                    ${(data?.responsibilities || ['']).map((resp, idx) => `
                        <div class="responsibility-item">
                            <input type="text" value="${resp}" placeholder="Built a feature that increased engagement by 40%">
                            <button class="btn-remove-item" onclick="removeResponsibility(this)">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <line x1="18" y1="6" x2="6" y2="18"/>
                                    <line x1="6" y1="6" x2="18" y2="18"/>
                                </svg>
                            </button>
                        </div>
                    `).join('')}
                </div>
                <button class="btn-add-item" onclick="addResponsibility('${id}_responsibilities')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="12" y1="5" x2="12" y2="19"/>
                        <line x1="5" y1="12" x2="19" y2="12"/>
                    </svg>
                    Add Responsibility
                </button>
            </div>
        </div>
    `;
    
    container.appendChild(card);
}

function addEducationItem(data = null) {
    const id = 'edu_' + Date.now();
    const container = document.getElementById('educationList');
    
    const card = document.createElement('div');
    card.className = 'item-card';
    card.id = id;
    
    card.innerHTML = `
        <div class="item-header" onclick="toggleCard('${id}')">
            <div class="item-title">${data?.degree || 'New Education'}</div>
            <div class="item-actions">
                <button class="btn-icon" onclick="event.stopPropagation(); toggleCard('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="18 15 12 9 6 15"/>
                    </svg>
                </button>
                <button class="btn-icon danger" onclick="event.stopPropagation(); removeItem('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        </div>
        <div class="item-fields">
            <div class="field-group">
                <label>Degree *</label>
                <input type="text" class="edu-degree" value="${data?.degree || ''}" placeholder="Bachelor of Science in Computer Science" required>
            </div>
            <div class="field-row">
                <div class="field-group">
                    <label>Institution *</label>
                    <input type="text" class="edu-institution" value="${data?.institution || ''}" placeholder="University Name" required>
                </div>
                <div class="field-group">
                    <label>Year</label>
                    <input type="text" class="edu-year" value="${data?.graduation_year || ''}" placeholder="2020">
                </div>
            </div>
            <div class="field-row">
                <div class="field-group">
                    <label>GPA</label>
                    <input type="text" class="edu-gpa" value="${data?.gpa || ''}" placeholder="3.8/4.0">
                </div>
                <div class="field-group">
                    <label>Location</label>
                    <input type="text" class="edu-location" value="${data?.location || ''}" placeholder="City, Country">
                </div>
            </div>
        </div>
    `;
    
    container.appendChild(card);
}

function addProjectItem(data = null) {
    const id = 'proj_' + Date.now();
    const container = document.getElementById('projectsList');
    
    const card = document.createElement('div');
    card.className = 'item-card';
    card.id = id;
    
    card.innerHTML = `
        <div class="item-header" onclick="toggleCard('${id}')">
            <div class="item-title">${data?.name || 'New Project'}</div>
            <div class="item-actions">
                <button class="btn-icon" onclick="event.stopPropagation(); toggleCard('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="18 15 12 9 6 15"/>
                    </svg>
                </button>
                <button class="btn-icon danger" onclick="event.stopPropagation(); removeItem('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        </div>
        <div class="item-fields">
            <div class="field-group">
                <label>Project Name *</label>
                <input type="text" class="proj-name" value="${data?.name || ''}" placeholder="E-commerce Platform" required>
            </div>
            <div class="field-group">
                <label>Description</label>
                <textarea class="proj-description" rows="3" placeholder="Built a scalable platform...">${data?.description || ''}</textarea>
            </div>
            <div class="field-group">
                <label>Technologies</label>
                <input type="text" class="proj-technologies" value="${(data?.technologies || []).join(', ')}" placeholder="React, Node.js, MongoDB">
            </div>
            <div class="field-group">
                <label>URL</label>
                <input type="url" class="proj-url" value="${data?.url || ''}" placeholder="https://project.com">
            </div>
        </div>
    `;
    
    container.appendChild(card);
}

function addCertificationItem(data = null) {
    const id = 'cert_' + Date.now();
    const container = document.getElementById('certificationsList');
    
    const card = document.createElement('div');
    card.className = 'item-card';
    card.id = id;
    
    card.innerHTML = `
        <div class="item-header" onclick="toggleCard('${id}')">
            <div class="item-title">${data?.name || 'New Certification'}</div>
            <div class="item-actions">
                <button class="btn-icon" onclick="event.stopPropagation(); toggleCard('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="18 15 12 9 6 15"/>
                    </svg>
                </button>
                <button class="btn-icon danger" onclick="event.stopPropagation(); removeItem('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        </div>
        <div class="item-fields">
            <div class="field-group">
                <label>Certification Name *</label>
                <input type="text" class="cert-name" value="${data?.name || ''}" placeholder="AWS Certified Solutions Architect" required>
            </div>
            <div class="field-row">
                <div class="field-group">
                    <label>Issuer</label>
                    <input type="text" class="cert-issuer" value="${data?.issuer || ''}" placeholder="Amazon Web Services">
                </div>
                <div class="field-group">
                    <label>Date</label>
                    <input type="text" class="cert-date" value="${data?.date || ''}" placeholder="Jan 2023">
                </div>
            </div>
            <div class="field-group">
                <label>URL</label>
                <input type="url" class="cert-url" value="${data?.url || ''}" placeholder="https://credential-url.com">
            </div>
        </div>
    `;
    
    container.appendChild(card);
}

function addAwardItem(data = null) {
    const id = 'award_' + Date.now();
    const container = document.getElementById('awardsList');
    
    const card = document.createElement('div');
    card.className = 'item-card';
    card.id = id;
    
    card.innerHTML = `
        <div class="item-header" onclick="toggleCard('${id}')">
            <div class="item-title">${data?.title || 'New Award'}</div>
            <div class="item-actions">
                <button class="btn-icon" onclick="event.stopPropagation(); toggleCard('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="18 15 12 9 6 15"/>
                    </svg>
                </button>
                <button class="btn-icon danger" onclick="event.stopPropagation(); removeItem('${id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        </div>
        <div class="item-fields">
            <div class="field-group">
                <label>Award Title *</label>
                <input type="text" class="award-title" value="${data?.title || ''}" placeholder="Employee of the Year" required>
            </div>
            <div class="field-row">
                <div class="field-group">
                    <label>Issuer</label>
                    <input type="text" class="award-issuer" value="${data?.issuer || ''}" placeholder="Company Name">
                </div>
                <div class="field-group">
                    <label>Date</label>
                    <input type="text" class="award-date" value="${data?.date || ''}" placeholder="2023">
                </div>
            </div>
            <div class="field-group">
                <label>Description</label>
                <textarea class="award-description" rows="2" placeholder="Recognized for outstanding performance...">${data?.description || ''}</textarea>
            </div>
        </div>
    `;
    
    container.appendChild(card);
}

// Helper Functions
function toggleCard(id) {
    const card = document.getElementById(id);
    card.classList.toggle('collapsed');
}

function removeItem(id) {
    if (confirm('Are you sure you want to remove this item?')) {
        document.getElementById(id).remove();
        showNotification('Item removed', 'success');
    }
}

function addResponsibility(listId) {
    const list = document.getElementById(listId);
    const item = document.createElement('div');
    item.className = 'responsibility-item';
    item.innerHTML = `
        <input type="text" placeholder="Describe your responsibility or achievement">
        <button class="btn-remove-item" onclick="removeResponsibility(this)">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
    `;
    list.appendChild(item);
}

function removeResponsibility(button) {
    button.parentElement.remove();
}

// Tags Functions
function populateTags(containerId, tags) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    tags.forEach(tag => addTag(containerId, tag));
}

function addTag(containerId, text) {
    const container = document.getElementById(containerId);
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.innerHTML = `
        <span>${text}</span>
        <span class="tag-remove" onclick="removeTag(this)">×</span>
    `;
    container.appendChild(tag);
}

function removeTag(element) {
    element.parentElement.remove();
}

function handleTagInput(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        const input = e.target;
        const text = input.value.trim();
        if (text) {
            const targetId = input.dataset.target;
            addTag(targetId, text);
            input.value = '';
        }
    }
}

function getTagsFromContainer(containerId) {
    const container = document.getElementById(containerId);
    const tags = container.querySelectorAll('.tag span:first-child');
    return Array.from(tags).map(tag => tag.textContent);
}

// Save Functions
function saveChanges() {
    try {
        // Update CV data with form values
        cvData.personal_info = {
            name: document.getElementById('name').value,
            email: document.getElementById('email').value,
            phone: document.getElementById('phone').value,
            location: document.getElementById('location').value,
            linkedin: document.getElementById('linkedin').value,
            github: document.getElementById('github').value,
            website: document.getElementById('website').value
        };
        
        cvData.professional_summary = document.getElementById('summary').value;
        
        // Save experience
        cvData.experience = [];
        document.querySelectorAll('#experienceList .item-card').forEach(card => {
            const responsibilities = [];
            card.querySelectorAll('.responsibility-item input').forEach(input => {
                if (input.value.trim()) responsibilities.push(input.value.trim());
            });
            
            cvData.experience.push({
                title: card.querySelector('.exp-title').value,
                company: card.querySelector('.exp-company').value,
                start_date: card.querySelector('.exp-start').value,
                end_date: card.querySelector('.exp-end').value,
                location: card.querySelector('.exp-location').value,
                responsibilities: responsibilities
            });
        });
        
        // Save education
        cvData.education = [];
        document.querySelectorAll('#educationList .item-card').forEach(card => {
            cvData.education.push({
                degree: card.querySelector('.edu-degree').value,
                institution: card.querySelector('.edu-institution').value,
                graduation_year: card.querySelector('.edu-year').value,
                gpa: card.querySelector('.edu-gpa').value,
                location: card.querySelector('.edu-location').value
            });
        });
        
        // Save skills
        cvData.skills = {
            programming_languages: getTagsFromContainer('programmingLanguages'),
            frameworks: getTagsFromContainer('frameworks'),
            tools: getTagsFromContainer('tools'),
            databases: getTagsFromContainer('databases'),
            cloud: getTagsFromContainer('cloud')
        };
        
        // Save projects
        cvData.projects = [];
        document.querySelectorAll('#projectsList .item-card').forEach(card => {
            const techInput = card.querySelector('.proj-technologies').value;
            const technologies = techInput ? techInput.split(',').map(t => t.trim()) : [];
            
            cvData.projects.push({
                name: card.querySelector('.proj-name').value,
                description: card.querySelector('.proj-description').value,
                technologies: technologies,
                url: card.querySelector('.proj-url').value
            });
        });
        
        // Save certifications
        cvData.certifications = [];
        document.querySelectorAll('#certificationsList .item-card').forEach(card => {
            cvData.certifications.push({
                name: card.querySelector('.cert-name').value,
                issuer: card.querySelector('.cert-issuer').value,
                date: card.querySelector('.cert-date').value,
                url: card.querySelector('.cert-url').value
            });
        });
        
        // Save awards
        cvData.awards = [];
        document.querySelectorAll('#awardsList .item-card').forEach(card => {
            cvData.awards.push({
                title: card.querySelector('.award-title').value,
                issuer: card.querySelector('.award-issuer').value,
                date: card.querySelector('.award-date').value,
                description: card.querySelector('.award-description').value
            });
        });
        
        // Save to localStorage
        localStorage.setItem('cvData', JSON.stringify(cvData));
        
        showModal();
        
    } catch (error) {
        console.error('Error saving changes:', error);
        showNotification('Error saving changes', 'error');
    }
}

function generateFinalCV() {
    saveChanges();
    showNotification('Generating final CV...', 'info');
    
    // Redirect to final CV generation page
    setTimeout(() => {
        window.location.href = 'final-cv.html';
    }, 1000);
}

// UI Functions
function showModal() {
    document.getElementById('successModal').style.display = 'flex';
    setTimeout(closeModal, 2000);
}

function closeModal() {
    document.getElementById('successModal').style.display = 'none';
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
        zIndex: '1001',
        animation: 'slideInRight 0.3s ease-out'
    });

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

console.log('✏️ Edit Template Page Ready!');
console.log('CV Data:', cvData);