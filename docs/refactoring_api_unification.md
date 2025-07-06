# API Unification Refactoring Plan

## Executive Summary

**Problem**: ForgeLLM currently has a confusing dual API architecture where model operations can be accessed through two different servers (Web Server port 5002 and Model Server port 5001), leading to inconsistencies, state synchronization issues, and maintenance overhead.

**Solution**: Unify into a single server architecture where all operations (CLI, Web UI, API clients) use the same consistent endpoints.

**Risk Level**: HIGH - This affects core functionality across the entire application.

**Timeline**: 3-4 days of focused work + extensive testing.

---

## Current Architecture Problems

### 1. Dual API Endpoints for Model Operations

**Problem**: Same operations accessible through different URLs with different behaviors.

```
Model Loading:
- CLI/Direct: http://localhost:5001/api/model/load  
- Web UI: http://localhost:5002/api/model/load → proxies to 5001

Model Generation:
- CLI/Direct: http://localhost:5001/api/model/generate (direct MLX)
- Web UI Streaming: http://localhost:5001/api/model/generate (direct)  
- Web UI Non-Streaming: http://localhost:5002/api/model/generate (proxy)
```

**Files Affected**:
- `forgellm/server/main.py` - Model Server (5001)
- `forgellm/api/routes.py` - Web Server (5002) 
- `forgellm/web/static/app.js` - Web UI client

### 2. State Synchronization Issues

**Problem**: Two separate state managers that can diverge.

**Model Server State** (`forgellm/server/main.py:40-44`):
```python
MODEL = None
TOKENIZER = None  
MODEL_NAME = None
ADAPTER_PATH = None
IS_LOADING = False
```

**ModelManager State** (`forgellm/models/model_manager.py:70-76`):
```python
self.model = None
self.tokenizer = None
self.model_name = None
self.adapter_path = None
self.loaded = False
self.loading = False
```

**Symptoms**: Adapter shows as loaded in Web UI but generation uses wrong model.

### 3. Different Generation Code Paths

**Model Server** (`forgellm/server/main.py:414+`):
- Direct MLX operations with `mlx_lm.generate.stream_generate`
- Sophisticated prompt formatting via `ModelArchitectureManager`
- Native streaming support

**ModelManager** (`forgellm/models/model_manager.py:390+`):
- Proxy to Model Server via HTTP requests
- Additional business logic layer
- No direct MLX access

### 4. Inconsistent Error Handling

**Example - Model Not Loaded**:
- Model Server: `{"success": false, "error": "No model loaded"}`
- Web Server: `{"success": False, "error": "No model loaded"}` (Python bool)

---

## Files Requiring Investigation

### Core Architecture Files
```
forgellm/
├── __main__.py                    # Entry point coordination
├── server/main.py                 # Model Server (port 5001)
├── api/routes.py                  # Web Server API (port 5002) 
├── models/model_manager.py        # Model state management
├── web/
│   ├── main.py                    # Web server startup
│   └── static/app.js             # Client-side API calls
└── cli/
    └── commands.py               # CLI commands
```

### Interdependency Analysis Required

**High Priority**:
1. **Training Pipeline** (`forgellm/training/`) - Does it depend on specific server?
2. **Model Publisher** (`forgellm/models/model_publisher.py`) - Publishing workflow
3. **Quantization** (`forgellm/models/model_quantizer.py`) - Model processing
4. **Dashboard** (`forgellm/training/dashboard.py`) - Real-time monitoring

**Medium Priority**:
5. **File Operations** - Model listing, directory browsing
6. **Configuration Management** - Environment variables, ports
7. **Process Management** - Server startup/shutdown coordination

**Low Priority**:
8. **Static Assets** - CSS, JS, templates
9. **Documentation** - API reference updates

---

## Proposed Unified Architecture

### Option A: Model Server as Primary (RECOMMENDED)

**Concept**: Extend Model Server (port 5001) to handle all operations.

```
┌─────────────┐    ┌─────────────────────────────────┐
│   CLI Tool  │────┤                                 │
├─────────────┤    │      Unified Server (5001)     │
│   Web UI    │────┤                                 │  
├─────────────┤    │  ┌─────────────┐ ┌─────────────┐│
│ API Clients │────┤  │   Model     │ │   Web       ││
└─────────────┘    │  │ Operations  │ │ Operations  ││
                   │  └─────────────┘ └─────────────┘│
                   └─────────────────────────────────┘
```

**Benefits**:
- Single source of truth for model state
- Direct MLX access for all clients
- Eliminates proxy overhead
- Simpler deployment (one server)

**Challenges**:
- Need to add web serving capabilities to Model Server
- Static file serving
- Session management for web UI

### Option B: Web Server as Primary

**Concept**: Move all MLX operations into Web Server, eliminate Model Server.

**Benefits**:
- Single server to manage
- All business logic in one place

**Challenges**:
- Need to refactor all MLX code from Model Server
- Higher complexity for CLI tools
- Potential memory issues (web server handling models)

---

## Detailed Refactoring Steps

### Phase 1: Investigation & Preparation (1 day)

#### 1.1 Map All API Calls
```bash
# Find all API endpoint usage
grep -r "localhost:5001" forgellm/
grep -r "localhost:5002" forgellm/
grep -r "/api/model/" forgellm/
grep -r "fetch.*api" forgellm/
```

#### 1.2 Identify Business Logic Dependencies
- **Training**: Does it call model APIs during training?
- **Quantization**: Does it load models via API?
- **Publishing**: How does it interact with models?

#### 1.3 Create Test Coverage Map
```bash
# Find what functionality is actually tested
grep -r "localhost:" tests/
grep -r "/api/" tests/
```

### Phase 2: Unified Server Implementation (2 days)

#### 2.1 Extend Model Server with Web Capabilities

**File**: `forgellm/server/main.py`

```python
# Add to ModelHandler class
def do_GET(self):
    if self.path.startswith('/api/'):
        # Existing API logic
    elif self.path.startswith('/static/'):
        # Serve static files  
    elif self.path == '/':
        # Serve main web UI
    else:
        # Handle other web routes
```

#### 2.2 Move Business Logic to Unified Server

**Current Web Server Routes** (`forgellm/api/routes.py`):
```python
@bp.route('/models', methods=['GET'])          # File operations
@bp.route('/training/start', methods=['POST']) # Training control  
@bp.route('/dashboard/data', methods=['GET'])  # Dashboard data
```

**Migration Strategy**:
- Move file operations to Model Server
- Keep training operations (or proxy them)
- Consolidate dashboard operations

#### 2.3 Update Client Code

**File**: `forgellm/web/static/app.js`

```javascript
// BEFORE (inconsistent)
await fetch('/api/model/load', {})              // → port 5002
await fetch('http://localhost:5001/api/model/generate', {})  // → port 5001

// AFTER (consistent)  
await fetch('/api/model/load', {})              // → port 5001
await fetch('/api/model/generate', {})          // → port 5001
```

### Phase 3: Eliminate Redundant Components (1 day)

#### 3.1 Remove Web Server

**Files to Remove**:
- `forgellm/api/routes.py` (move logic to Model Server)
- `forgellm/web/main.py` (update to start unified server)

#### 3.2 Simplify ModelManager

**File**: `forgellm/models/model_manager.py`

```python
# BEFORE: Proxy to separate server
def generate_text(self, params):
    response = requests.post(f"{self.server_url}/api/model/generate", ...)
    
# AFTER: Direct operations or simplified proxy
def generate_text(self, params):
    # Direct MLX operations or single consistent API call
```

#### 3.3 Update Entry Points

**File**: `forgellm/__main__.py`

```python
# BEFORE: Start both servers
def start_both_servers(args):
    # Start model server on 5001
    # Start web server on 5002
    
# AFTER: Start unified server  
def start_server(args):
    # Start unified server on configurable port (default 5001)
```

### Phase 4: Testing & Validation (1 day)

#### 4.1 Functional Testing
- Model loading/unloading
- Text generation (streaming & non-streaming)
- Adapter loading
- Training pipeline
- Web UI functionality

#### 4.2 Performance Testing
- Memory usage comparison
- Response time benchmarks
- Concurrent request handling

#### 4.3 Integration Testing
- CLI commands
- Web UI workflows  
- API client usage

---

## Risk Mitigation

### High Risk Areas

#### 1. Training Pipeline Breakage
**Risk**: Training may depend on specific server setup.
**Mitigation**: 
- Test training on simple model first
- Backup current working training code
- Gradual migration of training endpoints

#### 2. Model State Corruption  
**Risk**: Unified state management bugs.
**Mitigation**:
- Implement comprehensive state validation
- Add state persistence/recovery mechanisms
- Clear error messages for state issues

#### 3. Performance Regression
**Risk**: Single server handling both web and model operations.
**Mitigation**:
- Benchmark current performance
- Implement resource monitoring
- Consider async/threading for web operations

### Medium Risk Areas

#### 4. Configuration Management
**Risk**: Environment variables, port configurations become inconsistent.
**Mitigation**:
- Document all configuration changes
- Provide migration script for existing setups
- Maintain backward compatibility where possible

#### 5. Client Code Breakage  
**Risk**: External API clients may break.
**Mitigation**:
- Maintain API compatibility during transition
- Provide deprecation warnings
- Document migration path for external clients

---

## Migration Strategy

### Option 1: Big Bang Migration
- Stop everything, refactor, restart
- **Pros**: Clean, no transitional complexity
- **Cons**: High risk, long downtime for testing

### Option 2: Gradual Migration (RECOMMENDED)
1. **Phase 1**: Keep both servers, route all Web UI to Model Server
2. **Phase 2**: Move business logic to Model Server gradually  
3. **Phase 3**: Deprecate Web Server, update documentation
4. **Phase 4**: Remove Web Server code

### Option 3: Feature Flag Approach
- Add configuration flag to choose architecture
- Test unified server alongside existing one
- Switch when confident

---

## Success Criteria

### Functional Requirements
- [ ] All current Web UI functionality works
- [ ] All CLI commands work unchanged
- [ ] Model loading/generation identical behavior
- [ ] Training pipeline unchanged
- [ ] Adapter loading works correctly

### Non-Functional Requirements  
- [ ] No performance regression (< 10% slower)
- [ ] Memory usage similar or better
- [ ] Simpler deployment (one server instead of two)
- [ ] Easier debugging (single log stream)

### Documentation Requirements
- [ ] Updated API documentation
- [ ] Migration guide for existing users
- [ ] New architecture documentation
- [ ] Updated getting started guide

---

## Concrete Implementation Examples

### Example 1: Unified Model Loading

**Current** (`forgellm/api/routes.py:349`):
```python
@bp.route('/model/load', methods=['POST'])
def load_model():
    # Business logic + validation
    # Proxy to model_manager.load()
    # Return formatted response
```

**After Refactoring** (`forgellm/server/main.py`):
```python
def _handle_load(self, data):
    # All business logic here
    # Direct model loading
    # Consistent response format
```

### Example 2: File Operations Migration

**Current** (`forgellm/api/routes.py:1271`):
```python
@bp.route('/models', methods=['GET'])  
def get_models():
    # File system operations
    # Model metadata
    # Return JSON
```

**After** - Add to Model Server:
```python
def _handle_list_models(self):
    # Same logic, different location
    # Consistent with other endpoints
```

### Example 3: Client Code Simplification

**Current** (`forgellm/web/static/app.js:2284`):
```javascript
// Streaming: direct to model server
const response = await fetch('http://localhost:5001/api/model/generate', {

// Non-streaming: through web server  
const response = await fetch('/api/model/generate', {
```

**After**:
```javascript  
// All requests go to same server
const response = await fetch('/api/model/generate', {
```

---

## Emergency Rollback Plan

### Backup Strategy
1. Tag current working version in git
2. Document exact deployment steps
3. Keep model_server.py backup
4. Test rollback procedure on dev environment

### Rollback Triggers
- Any test failure during migration
- Performance regression > 20%
- Model loading/generation failures
- Training pipeline breakage
- Web UI functionality loss

### Rollback Steps
1. Stop unified server
2. Revert to tagged version
3. Start both servers as before
4. Validate all functionality
5. Document what went wrong

---

## Next Steps

1. **Review this plan** - Validate assumptions and approach
2. **Create feature branch** - `refactor/api-unification`
3. **Start with Phase 1** - Investigation and mapping
4. **Build comprehensive tests** - Before making any changes
5. **Implement gradually** - One phase at a time with validation

**Note**: This refactoring should not be rushed. Each phase should be thoroughly tested before proceeding to the next. 