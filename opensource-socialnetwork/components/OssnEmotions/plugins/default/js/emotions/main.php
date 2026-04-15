//<script>
/**
 * OssnEmotions - Client-side emotion detection integration
 * Connects OSSN UI to the Emotion API backend
 */
var OssnEmotions = OssnEmotions || {};

/**
 * API base URL (emo-app backend)
 */
OssnEmotions.apiUrl = 'http://localhost:8000/api/v1';

/**
 * AffectNet Image API base URL (facial emotion recognition)
 */
OssnEmotions.imageApiUrl = 'http://localhost:8000/image-api';

/**
 * Cache for analyzed posts to avoid re-fetching
 */
OssnEmotions.cache = {};

/**
 * Cache for image emotion analysis results
 */
OssnEmotions.imageCache = {};

/**
 * AffectNet face emotion → emoji mapping (8 emotions)
 */
OssnEmotions.faceEmojiMap = {
    'neutral':  '😐', 'happy':    '😊', 'sad':      '😢', 'surprise': '😲',
    'fear':     '😨', 'disgust':  '🤢', 'anger':    '😡', 'contempt': '😏'
};

/**
 * AffectNet face emotion → color mapping
 */
OssnEmotions.faceColorMap = {
    'neutral':  '#95A5A6', 'happy':    '#FFD93D', 'sad':      '#5B8FB9', 'surprise': '#F9CA24',
    'fear':     '#8854D0', 'disgust':  '#2ED573', 'anger':    '#FF4757', 'contempt': '#8B7E74'
};

/**
 * Emotion to emoji mapping
 */
OssnEmotions.emojiMap = {
    'joy': '😄', 'happy': '😊', 'love': '❤️', 'admiration': '🤩',
    'amusement': '😂', 'excitement': '🎉', 'gratitude': '🙏', 'optimism': '🌟',
    'pride': '💪', 'relief': '😌', 'caring': '🤗', 'approval': '👍',
    'desire': '😍', 'sadness': '😢', 'sad': '😢', 'grief': '😭',
    'disappointment': '😞', 'embarrassment': '😳', 'remorse': '😔',
    'anger': '😡', 'angry': '😡', 'annoyance': '😤', 'disapproval': '👎',
    'fear': '😨', 'nervousness': '😰', 'surprise': '😲', 'realization': '💡',
    'confusion': '😕', 'curiosity': '🤔', 'disgust': '🤢', 'neutral': '😐',
    'other': '🔵'
};

/**
 * Emotion to color mapping
 */
OssnEmotions.colorMap = {
    'joy': '#FFD93D', 'happy': '#FFD93D', 'love': '#FF6B6B', 'admiration': '#FF8FA3',
    'amusement': '#FFC93C', 'excitement': '#FF6F3C', 'gratitude': '#95E1D3',
    'optimism': '#7BE495', 'pride': '#DDA0DD', 'relief': '#87CEEB',
    'caring': '#FFB5B5', 'approval': '#98D8AA', 'desire': '#FF69B4',
    'sadness': '#5B8FB9', 'sad': '#5B8FB9', 'grief': '#4A6FA5',
    'disappointment': '#7B8FA1', 'embarrassment': '#C9A0DC', 'remorse': '#8B7E74',
    'anger': '#FF4757', 'angry': '#FF4757', 'annoyance': '#FF6348',
    'disapproval': '#E17055', 'fear': '#8854D0', 'nervousness': '#A29BFE',
    'surprise': '#F9CA24', 'realization': '#F0932B', 'confusion': '#FDCB6E',
    'curiosity': '#6C5CE7', 'disgust': '#2ED573', 'neutral': '#95A5A6',
    'other': '#BDC3C7'
};

/**
 * Analyze a wall post's text emotions
 */
OssnEmotions.analyzePost = function(guid) {
    var container = document.getElementById('emotion-badge-' + guid);
    if (!container) {
        // Create container if it doesn't exist
        var wallItem = document.getElementById('activity-item-' + guid);
        if (!wallItem) return;
        var commentsLikes = wallItem.querySelector('.comments-likes');
        if (!commentsLikes) return;
        container = document.createElement('div');
        container.className = 'ossn-emotion-badge-container';
        container.id = 'emotion-badge-' + guid;
        container.setAttribute('data-guid', guid);
        commentsLikes.parentNode.insertBefore(container, commentsLikes);
    }

    // Check cache
    if (OssnEmotions.cache[guid]) {
        OssnEmotions.renderBadge(guid, OssnEmotions.cache[guid]);
        return;
    }

    // Get post text
    var wallItem = document.getElementById('activity-item-' + guid);
    if (!wallItem) return;

    // Check if post has an image — analyze it too
    var hasImage = !!wallItem.querySelector('.ossn-wall-image-container img');
    if (hasImage) {
        OssnEmotions.analyzePostImage(guid);
    }
    
    var postContent = wallItem.querySelector('.post-contents p');
    if (!postContent || !postContent.textContent.trim()) {
        if (hasImage) {
            // Image-only post: text badge shows "Image post" instead of "No text to analyze"
            container.innerHTML = '<div class="ossn-emotion-badge"><span class="emotion-emoji">📸</span><span class="emotion-label" style="color:#6b7280">Image post — see facial analysis below</span></div>';
        } else {
            container.innerHTML = '<div class="ossn-emotion-badge"><span class="emotion-emoji">ℹ️</span><span class="emotion-label" style="color:#6b7280">No text to analyze</span></div>';
        }
        return;
    }

    var text = postContent.textContent.trim();
    
    // Show loading
    container.innerHTML = '<div class="ossn-emotion-loading"><div class="spinner"></div><span>Analyzing emotions...</span></div>';

    // Call enhanced analysis API
    OssnEmotions.callAPI('/text/enhanced', {
        text: text,
        include_emotions: true,
        include_sarcasm: true,
        include_slang: true,
        threshold: 0.3
    }, function(data) {
        OssnEmotions.cache[guid] = data;
        OssnEmotions.renderBadge(guid, data);
    }, function(error) {
        container.innerHTML = '<div class="ossn-emotion-badge"><span class="emotion-emoji">⚠️</span><span class="emotion-label" style="color:#dc2626">Analysis unavailable</span></div>';
    });
};

/**
 * Render the emotion badge on a post
 */
OssnEmotions.renderBadge = function(guid, data) {
    var container = document.getElementById('emotion-badge-' + guid);
    if (!container) return;

    var topEmotion = data.top_emotion || 'neutral';
    var emoji = OssnEmotions.emojiMap[topEmotion] || '🔵';
    var color = OssnEmotions.colorMap[topEmotion] || '#95A5A6';
    var sentiment = data.sentiment || 'neutral';
    
    // Calculate confidence percentage from emotions
    var confidence = 0;
    if (data.emotions && data.emotions.length > 0) {
        confidence = Math.round(data.emotions[0].probability * 100);
    }

    var html = '';
    
    // Main badge
    html += '<div class="ossn-emotion-badge" onclick="OssnEmotions.toggleDetail(' + guid + ')">';
    html += '<span class="emotion-emoji">' + emoji + '</span>';
    html += '<span class="emotion-label">' + OssnEmotions.capitalize(topEmotion) + '</span>';
    html += '<span class="emotion-confidence">' + confidence + '%</span>';
    html += '<span class="emotion-sentiment ' + sentiment + '">' + OssnEmotions.capitalize(sentiment) + '</span>';
    html += '</div>';

    // Detail panel (hidden by default)
    html += '<div class="ossn-emotion-detail-panel" id="emotion-detail-' + guid + '">';
    html += '<div class="ossn-emotion-detail-header">';
    html += '<h4>🧠 Emotion Analysis</h4>';
    html += '<button class="ossn-emotion-detail-close" onclick="OssnEmotions.toggleDetail(' + guid + ')">&times;</button>';
    html += '</div>';
    html += '<div class="ossn-emotion-detail-body">';

    // Emotion bars
    if (data.emotions && data.emotions.length > 0) {
        var topEmotions = data.emotions.slice(0, 6);
        for (var i = 0; i < topEmotions.length; i++) {
            var em = topEmotions[i];
            var pct = Math.round(em.probability * 100);
            var emColor = OssnEmotions.colorMap[em.emotion] || '#95A5A6';
            var emEmoji = OssnEmotions.emojiMap[em.emotion] || '🔵';
            
            html += '<div class="ossn-emotion-bar-container">';
            html += '<div class="ossn-emotion-bar-label">';
            html += '<span class="emotion-name">' + emEmoji + ' ' + OssnEmotions.capitalize(em.emotion) + '</span>';
            html += '<span class="emotion-pct">' + pct + '%</span>';
            html += '</div>';
            html += '<div class="ossn-emotion-bar-track">';
            html += '<div class="ossn-emotion-bar-fill" style="width:' + pct + '%;background:' + emColor + '"></div>';
            html += '</div>';
            html += '</div>';
        }
    }

    // Sarcasm alert
    if (data.sarcasm && data.sarcasm.is_sarcastic) {
        html += '<div class="ossn-emotion-sarcasm-alert">';
        html += '<span class="sarcasm-icon">🎭</span>';
        html += '<span><strong>Sarcasm detected</strong> (' + Math.round(data.sarcasm.confidence * 100) + '% confidence) — emotions may be inverted</span>';
        html += '</div>';
    }

    // Slang alert
    if (data.slang && data.slang.has_slang) {
        html += '<div class="ossn-emotion-slang-alert">';
        html += '<span class="sarcasm-icon">🗣️</span>';
        html += '<div>';
        html += '<strong>Slang detected:</strong> ';
        var terms = data.slang.slang_terms || [];
        html += terms.join(', ');
        if (data.slang.definitions) {
            html += '<br><small>';
            for (var term in data.slang.definitions) {
                html += '<em>' + term + '</em>: ' + data.slang.definitions[term] + '; ';
            }
            html += '</small>';
        }
        html += '</div>';
        html += '</div>';
    }

    // Emoji suggestions
    html += '<div class="ossn-emotion-emoji-panel">';
    html += '<h5>Suggested Reactions</h5>';
    html += '<div class="ossn-emotion-emoji-suggestions" id="emoji-suggest-' + guid + '">';
    html += '<span style="color:#6b7280;font-size:12px">Loading suggestions...</span>';
    html += '</div>';
    html += '</div>';

    // Recommendations
    if (data.recommendations && data.recommendations.length > 0) {
        html += '<div class="ossn-emotion-recommendations">';
        for (var r = 0; r < data.recommendations.length; r++) {
            var rec = data.recommendations[r];
            var recIcon = rec.type === 'warning' ? '⚠️' : (rec.type === 'info' ? 'ℹ️' : '💡');
            html += '<div class="ossn-emotion-recommendation-item">';
            html += '<span class="rec-icon">' + recIcon + '</span>';
            html += '<span>' + (rec.message || rec.text || JSON.stringify(rec)) + '</span>';
            html += '</div>';
        }
        html += '</div>';
    }

    html += '</div>'; // body
    html += '</div>'; // panel

    container.innerHTML = html;

    // Load emoji suggestions asynchronously
    OssnEmotions.loadEmojiSuggestions(guid);
};

/**
 * Toggle the detail panel
 */
OssnEmotions.toggleDetail = function(guid) {
    var panel = document.getElementById('emotion-detail-' + guid);
    if (panel) {
        panel.classList.toggle('active');
    }
};

/**
 * Load emoji suggestions for a post
 */
OssnEmotions.loadEmojiSuggestions = function(guid) {
    var wallItem = document.getElementById('activity-item-' + guid);
    if (!wallItem) return;
    
    var postContent = wallItem.querySelector('.post-contents p');
    if (!postContent || !postContent.textContent.trim()) return;
    
    OssnEmotions.callAPI('/emojis/suggest', {
        text: postContent.textContent.trim(),
        threshold: 0.3
    }, function(data) {
        var container = document.getElementById('emoji-suggest-' + guid);
        if (!container) return;
        
        var html = '';
        
        // Suggested emojis
        var suggested = data.suggested_emojis || [];
        for (var i = 0; i < Math.min(suggested.length, 12); i++) {
            html += '<span class="ossn-emotion-emoji-item" title="Suggested" onclick="OssnEmotions.copyEmoji(this)">' + suggested[i] + '</span>';
        }
        
        // Blocked emojis (show a few)
        var blocked = data.blocked_emojis || [];
        for (var i = 0; i < Math.min(blocked.length, 4); i++) {
            html += '<span class="ossn-emotion-emoji-item ossn-emotion-emoji-blocked" title="Inappropriate for this context">' + blocked[i] + '</span>';
        }
        
        container.innerHTML = html || '<span style="color:#6b7280;font-size:12px">No suggestions available</span>';
    }, function() {
        var container = document.getElementById('emoji-suggest-' + guid);
        if (container) {
            container.innerHTML = '<span style="color:#6b7280;font-size:12px">Suggestions unavailable</span>';
        }
    });
};

/**
 * Copy emoji to clipboard
 */
OssnEmotions.copyEmoji = function(element) {
    var emoji = element.textContent;
    if (navigator.clipboard) {
        navigator.clipboard.writeText(emoji).then(function() {
            element.style.transform = 'scale(1.5)';
            setTimeout(function() {
                element.style.transform = '';
            }, 200);
        });
    }
};

/**
 * Filter content before posting (hook into wall post form)
 */
OssnEmotions.filterBeforePost = function(text, callback) {
    if (!text || text.trim().length === 0) {
        callback(true, null);
        return;
    }

    OssnEmotions.callAPI('/filter/content', {
        text: text
    }, function(data) {
        if (data.should_block) {
            callback(false, data);
        } else {
            callback(true, data);
        }
    }, function() {
        // API unavailable, allow post through
        callback(true, null);
    });
};

/**
 * Auto-analyze text as user types (debounced)
 */
OssnEmotions._composeTimeout = null;
OssnEmotions.onComposeTyping = function(textarea) {
    clearTimeout(OssnEmotions._composeTimeout);
    OssnEmotions._composeTimeout = setTimeout(function() {
        var text = textarea.value.trim();
        var indicator = textarea.parentElement.querySelector('.ossn-emotion-compose-indicator');
        
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'ossn-emotion-compose-indicator';
            textarea.parentElement.appendChild(indicator);
        }

        if (text.length < 10) {
            indicator.classList.remove('active');
            return;
        }

        // Quick emotion check
        OssnEmotions.callAPI('/text/enhanced', {
            text: text,
            threshold: 0.3
        }, function(data) {
            var topEmotion = data.top_emotion || 'neutral';
            var emoji = OssnEmotions.emojiMap[topEmotion] || '🔵';
            var sentiment = data.sentiment || 'neutral';
            
            indicator.className = 'ossn-emotion-compose-indicator active';
            if (sentiment === 'negative') {
                indicator.className += ' warning';
            }
            indicator.innerHTML = emoji + ' <strong>' + OssnEmotions.capitalize(topEmotion) + '</strong> &middot; ' + OssnEmotions.capitalize(sentiment);
        }, function() {
            indicator.classList.remove('active');
        });
    }, 1500); // 1.5 second debounce
};

/**
 * Call the Emotion API
 */
OssnEmotions.callAPI = function(endpoint, data, onSuccess, onError) {
    var xhr = new XMLHttpRequest();
    xhr.open('POST', OssnEmotions.apiUrl + endpoint, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.timeout = 30000;

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    var response = JSON.parse(xhr.responseText);
                    if (onSuccess) onSuccess(response);
                } catch (e) {
                    if (onError) onError('Parse error');
                }
            } else {
                if (onError) onError('HTTP ' + xhr.status);
            }
        }
    };

    xhr.onerror = function() {
        if (onError) onError('Network error');
    };

    xhr.ontimeout = function() {
        if (onError) onError('Timeout');
    };

    xhr.send(JSON.stringify(data));
};

/**
 * Capitalize first letter
 */
OssnEmotions.capitalize = function(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
};

/**
 * Analyze a wall post's image for facial emotions via AffectNet API.
 * Fetches the image as a blob (same-origin) and sends to AffectNet /predict.
 * Falls back to PHP proxy if direct fetch fails.
 */
OssnEmotions.analyzePostImage = function(guid) {
    // Check cache
    if (OssnEmotions.imageCache[guid]) return;

    var wallItem = document.getElementById('activity-item-' + guid);
    if (!wallItem) return;

    var imgContainer = wallItem.querySelector('.ossn-wall-image-container img');
    if (!imgContainer || !imgContainer.src) return;

    var imgSrc = imgContainer.src;

    // Mark as in-progress to prevent duplicate calls
    OssnEmotions.imageCache[guid] = { _loading: true };

    // Try fetching the image as blob (same-origin) and POST to AffectNet
    var xhr = new XMLHttpRequest();
    xhr.open('GET', imgSrc, true);
    xhr.responseType = 'blob';
    xhr.timeout = 15000;

    xhr.onload = function() {
        if (xhr.status !== 200) {
            // Fallback to PHP proxy
            OssnEmotions.analyzePostImageViaProxy(guid);
            return;
        }

        var blob = xhr.response;
        var formData = new FormData();
        formData.append('file', blob, 'image.jpg');

        var apiXhr = new XMLHttpRequest();
        apiXhr.open('POST', OssnEmotions.imageApiUrl + '/predict', true);
        apiXhr.timeout = 60000;

        apiXhr.onreadystatechange = function() {
            if (apiXhr.readyState === 4) {
                if (apiXhr.status >= 200 && apiXhr.status < 300) {
                    try {
                        var data = JSON.parse(apiXhr.responseText);
                        OssnEmotions.imageCache[guid] = data;
                        OssnEmotions.renderImageBadge(guid, data);
                    } catch (e) {
                        delete OssnEmotions.imageCache[guid];
                    }
                } else {
                    // Fallback to PHP proxy
                    OssnEmotions.analyzePostImageViaProxy(guid);
                }
            }
        };

        apiXhr.onerror = function() {
            OssnEmotions.analyzePostImageViaProxy(guid);
        };

        apiXhr.ontimeout = function() {
            OssnEmotions.analyzePostImageViaProxy(guid);
        };

        apiXhr.send(formData);
    };

    xhr.onerror = function() {
        OssnEmotions.analyzePostImageViaProxy(guid);
    };

    xhr.send();
};

/**
 * Fallback: analyze image via PHP proxy action (server-side curl to AffectNet)
 */
OssnEmotions.analyzePostImageViaProxy = function(guid) {
    var token;
    try {
        token = typeof Ossn.Security !== 'undefined' ? Ossn.Security.token() : Ossn.getToken();
    } catch(e) {
        token = "";
    }
    
    var actionUrl = Ossn.site_url + 'action/emotion/analyze_image?ossn_ts=' +
        Ossn.session_token + '&ossn_token=' + token;

    var xhr = new XMLHttpRequest();
    xhr.open('POST', actionUrl, true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.timeout = 60000;

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    var resp = JSON.parse(xhr.responseText);
                    if (resp.success && resp.data) {
                        OssnEmotions.imageCache[guid] = resp.data;
                        OssnEmotions.renderImageBadge(guid, resp.data);
                    } else {
                        delete OssnEmotions.imageCache[guid];
                    }
                } catch (e) {
                    delete OssnEmotions.imageCache[guid];
                }
            } else {
                delete OssnEmotions.imageCache[guid];
            }
        }
    };

    xhr.onerror = function() { delete OssnEmotions.imageCache[guid]; };
    xhr.ontimeout = function() { delete OssnEmotions.imageCache[guid]; };

    xhr.send('guid=' + encodeURIComponent(guid));
};

/**
 * Render the image emotion badge on a wall post
 */
OssnEmotions.renderImageBadge = function(guid, data) {
    var wallItem = document.getElementById('activity-item-' + guid);
    if (!wallItem) return;

    // Find or create the image badge container (placed after the image)
    var containerId = 'image-emotion-badge-' + guid;
    var container = document.getElementById(containerId);
    if (!container) {
        var imgWrap = wallItem.querySelector('.ossn-wall-image-container');
        if (!imgWrap) return;
        container = document.createElement('div');
        container.className = 'ossn-emotion-badge-container ossn-image-emotion-badge-container';
        container.id = containerId;
        imgWrap.parentNode.insertBefore(container, imgWrap.nextSibling);
    }

    var emotion = data.predicted_emotion || 'neutral';
    var confidence = Math.round((data.confidence || 0) * 100);
    var emoji = OssnEmotions.faceEmojiMap[emotion] || '🔵';
    var color = OssnEmotions.faceColorMap[emotion] || '#95A5A6';
    var probs = data.all_probabilities || {};

    var html = '';

    // Main badge with camera icon to distinguish from text badge
    html += '<div class="ossn-emotion-badge ossn-image-emotion-badge" onclick="OssnEmotions.toggleDetail(\'img-' + guid + '\')">';
    html += '<span class="emotion-source-icon">📸</span>';
    html += '<span class="emotion-emoji">' + emoji + '</span>';
    html += '<span class="emotion-label">' + OssnEmotions.capitalize(emotion) + '</span>';
    html += '<span class="emotion-confidence">' + confidence + '%</span>';
    html += '</div>';

    // Detail panel
    html += '<div class="ossn-emotion-detail-panel" id="emotion-detail-img-' + guid + '">';
    html += '<div class="ossn-emotion-detail-header ossn-image-emotion-header">';
    html += '<h4>📸 Facial Emotion Analysis</h4>';
    html += '<button class="ossn-emotion-detail-close" onclick="OssnEmotions.toggleDetail(\'img-' + guid + '\')">&times;</button>';
    html += '</div>';
    html += '<div class="ossn-emotion-detail-body">';

    // Sort probabilities descending
    var sortedEmotions = Object.keys(probs).sort(function(a, b) { return probs[b] - probs[a]; });

    for (var i = 0; i < sortedEmotions.length; i++) {
        var em = sortedEmotions[i];
        var pct = Math.round(probs[em] * 100);
        var emColor = OssnEmotions.faceColorMap[em] || '#95A5A6';
        var emEmoji = OssnEmotions.faceEmojiMap[em] || '🔵';

        html += '<div class="ossn-emotion-bar-container">';
        html += '<div class="ossn-emotion-bar-label">';
        html += '<span class="emotion-name">' + emEmoji + ' ' + OssnEmotions.capitalize(em) + '</span>';
        html += '<span class="emotion-pct">' + pct + '%</span>';
        html += '</div>';
        html += '<div class="ossn-emotion-bar-track">';
        html += '<div class="ossn-emotion-bar-fill" style="width:' + pct + '%;background:' + emColor + '"></div>';
        html += '</div>';
        html += '</div>';
    }

    html += '</div>'; // body
    html += '</div>'; // panel

    container.innerHTML = html;
};

/**
 * Get image emotion data for a post (for reaction filtering)
 * Returns null if no image data is available or still loading
 */
OssnEmotions.getImageEmotionForPost = function(guid) {
    var cached = OssnEmotions.imageCache[guid];
    if (!cached || cached._loading) return null;
    return cached;
};

/**
 * Initialize: attach to existing wall posts, compose box, and reaction filtering
 * IMPORTANT: Must wrap in $(document).ready() because Ossn.Init() runs in <head>
 * before the DOM body exists. Without this, querySelectorAll, $('body').on(),
 * and MutationObserver on document.body all silently fail.
 */
Ossn.RegisterStartupFunction(function() {
    $(document).ready(function() {

    // ======================================================================
    // PRE-ANALYZE ALL POSTS on page load for reaction filtering
    // This ensures reaction data is cached BEFORE user hovers Like
    // ======================================================================
    var wallItems = document.querySelectorAll('.ossn-wall-item');

    // Pre-analyze each post's text for reaction filtering (lightweight call)
    for (var i = 0; i < wallItems.length; i++) {
        (function(item) {
            var guid = item.id.replace('activity-item-', '');
            if (!guid || isNaN(guid)) return;

            var postContent = item.querySelector('.post-contents p');
            var hasText = postContent && postContent.textContent.trim();
            var hasImage = !!item.querySelector('.ossn-wall-image-container img');

            // Pre-analyze text for reaction filtering
            if (hasText) {
                OssnEmotions.callAPI('/text/enhanced', {
                    text: postContent.textContent.trim(),
                    threshold: 0.3
                }, function(data) {
                    OssnEmotions.cache[guid] = data;
                }, function() {});
            }

            // Pre-analyze image for reaction filtering (especially image-only posts)
            if (hasImage) {
                OssnEmotions.analyzePostImage(guid);
            }
        })(wallItems[i]);
    }

    // Also run the full analysis with badges for visible posts (first 5)
    var analyzed = 0;
    for (var i = 0; i < wallItems.length && analyzed < 5; i++) {
        var item = wallItems[i];
        var guid = item.id.replace('activity-item-', '');
        if (guid && !isNaN(guid)) {
            OssnEmotions.analyzePost(parseInt(guid));
            analyzed++;
        }
    }

    // Attach to compose textareas for live emotion preview
    var composeAreas = document.querySelectorAll('.ossn-wall-container textarea, .ossn-wall-container-data textarea');
    for (var j = 0; j < composeAreas.length; j++) {
        composeAreas[j].addEventListener('input', function() {
            OssnEmotions.onComposeTyping(this);
        });
    }

    // Hook into wall post AJAX: analyze new posts when they appear
    var activityContainer = document.querySelector('.user-activity');
    if (activityContainer) {
        var observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1 && node.classList && node.classList.contains('ossn-wall-item')) {
                            var newGuid = node.id.replace('activity-item-', '');
                            if (newGuid && !isNaN(newGuid)) {
                                setTimeout(function() {
                                    // Pre-analyze text for reactions
                                    var pc = node.querySelector('.post-contents p');
                                    if (pc && pc.textContent.trim()) {
                                        OssnEmotions.callAPI('/text/enhanced', {
                                            text: pc.textContent.trim(),
                                            threshold: 0.3
                                        }, function(data) {
                                            OssnEmotions.cache[newGuid] = data;
                                        }, function() {});
                                    }
                                    // Pre-analyze image for reactions
                                    if (node.querySelector('.ossn-wall-image-container img')) {
                                        OssnEmotions.analyzePostImage(newGuid);
                                    }
                                    // Full analysis with badge
                                    OssnEmotions.analyzePost(parseInt(newGuid));
                                }, 500);
                            }
                        }
                    });
                }
            });
        });
        observer.observe(activityContainer, { childList: true });
    }

    // ======================================================================
    // REACTION FILTERING: Only show relevant reactions based on post emotion
    // Uses monkey-patching of $MenuReactions for synchronous, reliable filtering
    // ======================================================================

    // Map OSSN reaction CSS classes to reaction names
    var reactionMapping = {
        'ossn-like-reaction-like':    'like',
        'ossn-like-reaction-dislike': 'dislike',
        'ossn-like-reaction-love':    'love',
        'ossn-like-reaction-haha':    'haha',
        'ossn-like-reaction-yay':     'yay',
        'ossn-like-reaction-wow':     'wow',
        'ossn-like-reaction-sad':     'sad',
        'ossn-like-reaction-angry':   'angry'
    };

    // Emotion-to-reaction mapping: which reactions make sense for each detected emotion
    var emotionToReactions = {
        // Positive emotions
        'joy':           ['like', 'love', 'haha', 'yay'],
        'happy':         ['like', 'love', 'haha', 'yay'],
        'love':          ['like', 'love', 'yay'],
        'admiration':    ['like', 'love', 'wow', 'yay'],
        'amusement':     ['like', 'haha', 'yay'],
        'excitement':    ['like', 'love', 'haha', 'yay', 'wow'],
        'gratitude':     ['like', 'love', 'yay'],
        'optimism':      ['like', 'love', 'yay'],
        'pride':         ['like', 'love', 'wow', 'yay'],
        'relief':        ['like', 'love', 'yay'],
        'caring':        ['like', 'love', 'yay'],
        'approval':      ['like', 'love', 'yay'],
        'desire':        ['like', 'love', 'wow'],
        // Negative emotions
        'sadness':       ['like', 'love', 'sad'],
        'sad':           ['like', 'love', 'sad'],
        'grief':         ['like', 'love', 'sad'],
        'disappointment':['like', 'sad'],
        'embarrassment': ['like', 'love', 'sad', 'wow'],
        'remorse':       ['like', 'love', 'sad'],
        'anger':         ['like', 'angry', 'wow', 'sad'],
        'angry':         ['like', 'angry', 'wow', 'sad'],
        'annoyance':     ['like', 'angry', 'sad'],
        'disapproval':   ['like', 'dislike', 'angry', 'sad'],
        // Other
        'fear':          ['like', 'love', 'wow', 'sad'],
        'nervousness':   ['like', 'love', 'wow', 'sad'],
        'surprise':      ['like', 'wow', 'haha'],
        'realization':   ['like', 'wow'],
        'confusion':     ['like', 'wow'],
        'curiosity':     ['like', 'wow'],
        'disgust':       ['like', 'angry', 'wow'],
        'neutral':       ['like', 'love', 'haha', 'yay', 'wow', 'sad', 'angry', 'dislike'],
        'other':         ['like', 'love', 'haha', 'yay', 'wow', 'sad', 'angry', 'dislike']
    };

    // AffectNet face emotion → reaction mapping (8 facial emotions)
    var faceEmotionToReactions = {
        'neutral':  ['like', 'love', 'haha', 'yay', 'wow', 'sad', 'angry', 'dislike'],
        'happy':    ['like', 'love', 'haha', 'yay'],
        'sad':      ['like', 'love', 'sad'],
        'surprise': ['like', 'wow', 'haha', 'yay'],
        'fear':     ['like', 'love', 'wow', 'sad'],
        'disgust':  ['like', 'angry', 'wow', 'dislike'],
        'anger':    ['like', 'angry', 'wow', 'sad'],
        'contempt': ['like', 'dislike', 'angry', 'wow']
    };

    // Cache for reaction filter results per post guid
    var reactionFilterCache = {};

    /**
     * Determine allowed reactions from emotion analysis data
     */
    function getAllowedReactions(emotionData) {
        var allowedSet = {};
        allowedSet['like'] = true;

        var topEmotion = emotionData.top_emotion || 'neutral';
        var emotions = emotionData.emotions || [];

        var topRules = emotionToReactions[topEmotion] || emotionToReactions['neutral'];
        for (var i = 0; i < topRules.length; i++) {
            allowedSet[topRules[i]] = true;
        }

        for (var j = 0; j < Math.min(emotions.length, 3); j++) {
            if (emotions[j].probability > 0.2) {
                var secRules = emotionToReactions[emotions[j].emotion] || [];
                for (var k = 0; k < secRules.length; k++) {
                    allowedSet[secRules[k]] = true;
                }
            }
        }

        return allowedSet;
    }

    /**
     * Determine allowed reactions from AffectNet image emotion data
     */
    function getAllowedReactionsFromImage(imageData) {
        var allowedSet = {};
        allowedSet['like'] = true;

        var topEmotion = imageData.predicted_emotion || 'neutral';
        var probs = imageData.all_probabilities || {};

        // Primary emotion reactions
        var topRules = faceEmotionToReactions[topEmotion] || faceEmotionToReactions['neutral'];
        for (var i = 0; i < topRules.length; i++) {
            allowedSet[topRules[i]] = true;
        }

        // Also consider secondary emotions with >20% probability
        for (var em in probs) {
            if (probs[em] > 0.2 && em !== topEmotion) {
                var secRules = faceEmotionToReactions[em] || [];
                for (var k = 0; k < secRules.length; k++) {
                    allowedSet[secRules[k]] = true;
                }
            }
        }

        return allowedSet;
    }

    /**
     * Apply filtering to a reaction panel's <li> elements
     */
    function applyReactionFilter(panel, allowedReactions, topEmotion) {
        var $panel = $(panel);
        if ($panel.data('emotion-filtered')) return;
        $panel.data('emotion-filtered', true);

        $panel.find('li').each(function() {
            var $reaction = $(this);
            var classes = $reaction.attr('class') || '';

            for (var className in reactionMapping) {
                if (classes.indexOf(className) !== -1) {
                    var reactionName = reactionMapping[className];
                    if (!allowedReactions[reactionName]) {
                        $reaction.addClass('emotion-reaction-blocked');
                        $reaction.attr('title', 'Not relevant for this post');
                        $reaction.removeAttr('onclick');
                        $reaction.removeAttr('href');
                        $reaction.removeAttr('data-reaction');
                        $reaction.off('click');
                        $reaction.on('click', function(e) { e.preventDefault(); e.stopPropagation(); return false; });
                    } else {
                        $reaction.addClass('emotion-reaction-allowed');
                    }
                    break;
                }
            }
        });

        if (topEmotion && topEmotion !== 'neutral') {
            var emojiIcon = OssnEmotions.emojiMap[topEmotion] || OssnEmotions.faceEmojiMap[topEmotion] || '';
            if (emojiIcon && $panel.find('.reaction-emotion-tag').length === 0) {
                $panel.append('<span class="reaction-emotion-tag" title="Post emotion: ' +
                    OssnEmotions.capitalize(topEmotion) + '">' + emojiIcon + '</span>');
            }
        }
    }

    /**
     * Get cached filter data for a post guid
     * Falls back to image emotion data when text data is unavailable
     */
    function getFilterDataForGuid(guid) {
        if (reactionFilterCache[guid]) return reactionFilterCache[guid];

        // Try text emotion cache first
        if (OssnEmotions.cache[guid]) {
            var allowed = getAllowedReactions(OssnEmotions.cache[guid]);
            var topEm = OssnEmotions.cache[guid].top_emotion || 'neutral';
            reactionFilterCache[guid] = { allowed: allowed, topEmotion: topEm, source: 'text' };
            return reactionFilterCache[guid];
        }

        // Fall back to image emotion cache (for image-only posts)
        var imgData = OssnEmotions.getImageEmotionForPost(guid);
        if (imgData) {
            var allowed = getAllowedReactionsFromImage(imgData);
            var topEm = imgData.predicted_emotion || 'neutral';
            reactionFilterCache[guid] = { allowed: allowed, topEmotion: topEm, source: 'image' };
            return reactionFilterCache[guid];
        }

        return null;
    }

    /**
     * Filter a reaction panel: use cached data or call API.
     * For image-only posts, uses face emotion from AffectNet.
     */
    function filterReactionPanel(panel) {
        var wallItem = $(panel).closest('.ossn-wall-item');
        if (wallItem.length === 0) return;

        var guid = wallItem.attr('id').replace('activity-item-', '');
        if (!guid || isNaN(guid)) return;

        var filterData = getFilterDataForGuid(guid);
        if (filterData) {
            applyReactionFilter(panel, filterData.allowed, filterData.topEmotion);
            return;
        }

        // No cached data — try text API first
        var postContent = wallItem.find('.post-contents p');
        var hasText = postContent.length > 0 && postContent.text().trim();
        var hasImage = wallItem.find('.ossn-wall-image-container img').length > 0;

        if (hasText) {
            // Text post: call text emotion API
            $(panel).addClass('emotion-reaction-loading');
            $(panel).find('li').css('visibility', 'hidden');

            OssnEmotions.callAPI('/text/enhanced', {
                text: postContent.text().trim(),
                threshold: 0.3
            }, function(data) {
                OssnEmotions.cache[guid] = data;
                var allowed = getAllowedReactions(data);
                var topEm = data.top_emotion || 'neutral';
                reactionFilterCache[guid] = { allowed: allowed, topEmotion: topEm, source: 'text' };

                $(panel).removeClass('emotion-reaction-loading');
                $(panel).find('li').css('visibility', '');
                applyReactionFilter(panel, allowed, topEm);
            }, function() {
                $(panel).removeClass('emotion-reaction-loading');
                $(panel).find('li').css('visibility', '');
            });
        } else if (hasImage) {
            // Image-only post: trigger image analysis and wait for result
            $(panel).addClass('emotion-reaction-loading');
            $(panel).find('li').css('visibility', 'hidden');

            // Start image analysis if not already started
            OssnEmotions.analyzePostImage(guid);

            // Poll for image result (analysis is async)
            var pollCount = 0;
            var pollInterval = setInterval(function() {
                pollCount++;
                var imgData = OssnEmotions.getImageEmotionForPost(guid);
                if (imgData) {
                    clearInterval(pollInterval);
                    var allowed = getAllowedReactionsFromImage(imgData);
                    var topEm = imgData.predicted_emotion || 'neutral';
                    reactionFilterCache[guid] = { allowed: allowed, topEmotion: topEm, source: 'image' };

                    $(panel).removeClass('emotion-reaction-loading');
                    $(panel).find('li').css('visibility', '');
                    applyReactionFilter(panel, allowed, topEm);
                } else if (pollCount > 30) {
                    // Timeout after ~15 seconds — show all reactions unfiltered
                    clearInterval(pollInterval);
                    $(panel).removeClass('emotion-reaction-loading');
                    $(panel).find('li').css('visibility', '');
                }
            }, 500);
        }
        // If neither text nor image, reactions remain unfiltered
    }

    // ======================================================================
    // MONKEY-PATCH $MenuReactions for synchronous, reliable reaction filtering
    // $MenuReactions is defined by OssnLikes inside its own $(document).ready().
    // We use setTimeout(0) to ensure our patch runs AFTER OssnLikes defines it.
    // ======================================================================
    setTimeout(function() {
        if (typeof $MenuReactions === 'function') {
            var _originalMenuReactions = $MenuReactions;
            $MenuReactions = function($elem) {
                // Call original to create the reaction panel
                var result = _originalMenuReactions($elem);

                // Find the panel that was just created in the parent
                var $parent = $($elem).parent();
                var $panel = $parent.find('.ossn-like-reactions-panel');
                if ($panel.length > 0 && !$panel.data('emotion-filtered')) {
                    filterReactionPanel($panel[0]);
                }

                return result;
            };
        }
    }, 0);

    // Comment reactions don't use $MenuReactions, so handle separately
    $('body').on('mouseenter touchstart', '.ossn-like-comment', function() {
        var $btn = $(this);
        setTimeout(function() {
            var $panel = $btn.parent().parent().find('.ossn-like-reactions-panel');
            if ($panel.length > 0 && !$panel.data('emotion-filtered')) {
                filterReactionPanel($panel[0]);
            }
        }, 150);
    });

    }); // end $(document).ready()
});

/**
 * ============================================================
 * MoodBuddy – Emotional Wellness Chat Assistant
 * ============================================================
 */
Ossn.RegisterStartupFunction(function() {
$(document).ready(function() {

    var $widget   = $('#moodbuddy-widget');
    if (!$widget.length) return; // widget not rendered (not logged in)

    var $trigger  = $('#moodbuddy-trigger');
    var $chat     = $('#moodbuddy-chat');
    var $close    = $('#moodbuddy-close');
    var $messages = $('#moodbuddy-messages');
    var $input    = $('#moodbuddy-input');
    var $send     = $('#moodbuddy-send');
    var $emotionBar = $('#moodbuddy-emotion-bar');
    var $emotionTag = $('#moodbuddy-emotion-tag');

    var conversationHistory = [];
    var isOpen = false;
    var isSending = false;

    // Emotion emoji map for detected emotions
    var emotionEmojis = {
        admiration: '🌟', amusement: '😄', anger: '😤', annoyance: '😒',
        approval: '👍', caring: '💗', confusion: '😕', curiosity: '🤔',
        desire: '💫', disappointment: '😞', disapproval: '👎', disgust: '🤢',
        embarrassment: '😳', excitement: '🎉', fear: '😨', gratitude: '🙏',
        grief: '😢', joy: '😊', love: '❤️', nervousness: '😰',
        optimism: '🌈', pride: '🦁', realization: '💡', relief: '😌',
        remorse: '😔', sadness: '💙', surprise: '😲', neutral: '😐'
    };

    // Toggle chat window
    $trigger.on('click', function() {
        if (isOpen) {
            closeChat();
        } else {
            openChat();
        }
    });

    $close.on('click', function() {
        closeChat();
    });

    function openChat() {
        $chat.slideDown(200);
        $trigger.addClass('moodbuddy-trigger-active');
        isOpen = true;
        $input.focus();
        scrollToBottom();
    }

    function closeChat() {
        $chat.slideUp(200);
        $trigger.removeClass('moodbuddy-trigger-active');
        isOpen = false;
    }

    // Send on click
    $send.on('click', function() {
        sendMessage();
    });

    // Send on Enter (Shift+Enter for newline)
    $input.on('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    $input.on('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });

    function sendMessage() {
        var message = $input.val().trim();
        if (!message || isSending) return;

        isSending = true;
        $send.prop('disabled', true);
        $input.val('').css('height', 'auto');

        // Display user message
        appendMessage('user', message);

        // Show typing indicator
        var typingId = showTyping();

        // Build history for context (last 10 messages)
        var historyToSend = conversationHistory.slice(-10);

        // Get emotion context from recent posts on the page
        var emotionContext = getPageEmotionContext();

        // Send to backend via OSSN action using Ossn.PostRequest (handles tokens automatically)
        var params = 'message=' + encodeURIComponent(message) +
                     '&conversation_history=' + encodeURIComponent(JSON.stringify(historyToSend));
        if (emotionContext) {
            params += '&emotion_context=' + encodeURIComponent(emotionContext);
        }

        Ossn.PostRequest({
            url: Ossn.site_url + 'action/emotion/chat',
            params: params,
            callback: function(data) {
                removeTyping(typingId);
                isSending = false;
                $send.prop('disabled', false);
                $input.focus();

                // Parse if string
                if (typeof data === 'string') {
                    try { data = JSON.parse(data); } catch(e) {
                        appendMessage('bot', 'Hmm, something went wrong. Could you try again? 💙');
                        return;
                    }
                }

                if (data && data.response) {
                    appendMessage('bot', data.response);

                    // Store in history
                    conversationHistory.push({ role: 'user', content: message });
                    conversationHistory.push({ role: 'assistant', content: data.response });

                    // Show emotion if detected
                    if (data.detected_emotion) {
                        showEmotion(data.detected_emotion);
                    } else {
                        $emotionBar.slideUp(150);
                    }
                } else {
                    appendMessage('bot', 'Hmm, something went wrong. Could you try again? 💙');
                }
            },
            error: function(xhr, status, error) {
                removeTyping(typingId);
                isSending = false;
                $send.prop('disabled', false);
                $input.focus();
                console.error('MoodBuddy error:', status, error);
                appendMessage('bot', 'I\'m having trouble connecting. Please check if the service is running and try again! 💙');
            }
        });
    }

    function appendMessage(type, text) {
        var isBot = (type === 'bot');
        var avatarHtml = isBot ? '<div class="moodbuddy-message-avatar">🧠</div>' : '';
        var cssClass = isBot ? 'moodbuddy-bot' : 'moodbuddy-user';

        // Sanitize text to prevent XSS
        var safeText = $('<div>').text(text).html();
        // Allow basic safe formatting: convert newlines and emojis
        safeText = safeText.replace(/\n/g, '<br>');

        var html = '<div class="moodbuddy-message ' + cssClass + '">' +
                       avatarHtml +
                       '<div class="moodbuddy-message-bubble">' + safeText + '</div>' +
                   '</div>';
        $messages.append(html);
        scrollToBottom();
    }

    function showTyping() {
        var id = 'typing-' + Date.now();
        var html = '<div class="moodbuddy-message moodbuddy-bot moodbuddy-typing" id="' + id + '">' +
                       '<div class="moodbuddy-message-avatar">🧠</div>' +
                       '<div class="moodbuddy-message-bubble">' +
                           '<div class="moodbuddy-dots">' +
                               '<span></span><span></span><span></span>' +
                           '</div>' +
                       '</div>' +
                   '</div>';
        $messages.append(html);
        scrollToBottom();
        return id;
    }

    function removeTyping(id) {
        $('#' + id).remove();
    }

    function showEmotion(emotion) {
        var emoji = emotionEmojis[emotion] || '😐';
        $emotionTag.text(emoji + ' Feeling: ' + emotion);
        $emotionBar.slideDown(150);
    }

    function scrollToBottom() {
        var el = $messages[0];
        if (el) el.scrollTop = el.scrollHeight;
    }

    function getPageEmotionContext() {
        // Summarize emotions from recent posts on the page for context
        var emotions = [];
        var keys = Object.keys(OssnEmotions.cache || {});
        for (var i = 0; i < Math.min(keys.length, 3); i++) {
            var data = OssnEmotions.cache[keys[i]];
            if (data && data.top_emotion) {
                emotions.push(data.top_emotion);
            }
        }
        if (emotions.length > 0) {
            return 'Recent post emotions on feed: ' + emotions.join(', ');
        }
        return null;
    }

}); // end $(document).ready
}); // end RegisterStartupFunction

