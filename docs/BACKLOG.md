# ForgeLL Backlog

## Performance & Optimization

### 📡 Session Badge Population API Optimization

**Issue**: Potential for API request floods during session card population in UI tabs

**Background**: 
The `populateLossBadges()` function can trigger many simultaneous API requests when loading session lists:

**Current Triggers**:
1. **Compare Tab**: `loadSessions()` calls `populateLossBadges(sessions)` with all sessions (50+)
2. **Training Tab**: `updateTrainingSessionsList()` calls `populateLossBadges(trainingSessions)` 
3. **Tab Activation**: Badge population triggered on Compare tab activation if badges not populated
4. **Session Restoration**: Potential indirect triggers during session restoration

**Current Protection**:
- ✅ Training detection blocks badge population during active training
- ✅ Rate limiting (10 requests/5 seconds) as backup protection

**Improvement Opportunities**:

#### 1. Batch API Requests
- **Current**: 1 API call per session (`/api/dashboard/historical` for each session)
- **Proposed**: Single batch API endpoint accepting multiple session IDs
- **Benefit**: 50+ individual requests → 1 batch request

#### 2. Smart Caching Strategy  
- **Current**: No caching between tab switches
- **Proposed**: Session-level caching with TTL (time-to-live)
- **Benefit**: Avoid re-fetching same session data repeatedly

#### 3. Progressive Loading
- **Current**: Load all session badges simultaneously
- **Proposed**: Load badges progressively (visible sessions first, others on scroll)
- **Benefit**: Better perceived performance, reduced initial load

#### 4. API Response Optimization
- **Current**: Full session data including charts for badge population
- **Proposed**: Lightweight badge-only endpoint returning just loss values
- **Benefit**: Reduced memory usage and faster response times

**Priority**: Medium (performance improvement, not critical functionality)

**Implementation Complexity**: Medium (requires API changes and caching logic)

---

## UI/UX Improvements

### 🎨 Session Card Performance

**Issue**: Large session lists (50+ sessions) can cause UI lag during initial render

**Proposed Solutions**:
- Virtual scrolling for session lists
- Lazy loading of session card content
- Debounced search/filter operations

**Priority**: Low

---

## Technical Debt

### 🔄 API Consolidation

**Issue**: Multiple similar endpoints for session data

**Current APIs**:
- `/api/training/sessions` - Basic session list
- `/api/dashboard/historical` - Full session data with charts
- `/api/logs/raw` - Raw session logs

**Proposed**: Unified session API with query parameters for different data levels

**Priority**: Low 