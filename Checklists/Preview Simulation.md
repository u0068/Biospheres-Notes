# Implementation Checklist

## Overview

The Preview simulation is a fast, deterministic mode designed for genome editing and testing. It runs on a single thread to ensure perfect reproducibility and updates in real-time as genomes are modified.

**Core Requirements:**

- **Cell Count:** 256 cells maximum
- **Performance Target:** 2Fr (256 cells × 30 ticks in 16ms)
- **Execution:** Single-threaded for bit-for-bit determinism
- **Control:** Manual time scrubber only
- **Primary Use:** Genome editing and quick organism testing

---

## Table of Contents

1. [[#Core Architecture]]
2. [[#Physics Systems]]
3. [[#Cell Types Implementation]]
4. [[#Genome System Integration]]
5. [[#Rendering System]]
6. [[#UI Integration]]
7. [[#Performance Optimization]]
8. [[#Testing & Validation]]

---

## Core Architecture

### Data Structures

#### Cell Data (Structure of Arrays)

- [ ] Position array (`[Vec3; 256]`)
- [ ] Velocity array (`[Vec3; 256]`)
- [ ] Rotation array (`[Quat; 256]`)
- [ ] Angular velocity array (`[Vec3; 256]`)
- [ ] Mass array (`[f32; 256]`)
- [ ] Radius array (`[f32; 256]`)
- [ ] Age array (`[f32; 256]`)
- [ ] Split count array (`[u32; 256]`)
- [ ] Genome index array (`[u32; 256]`)
- [ ] Mode index array (`[u8; 256]`)
- [ ] Cell type array (`[u8; 256]`)
- [ ] Active flag array (`[bool; 256]`)
- [ ] Cell count tracker (`usize`)

#### Adhesion Data

- [ ] Adhesion array with fixed capacity (e.g., `[Adhesion; 1280]` for ~10 per cell max)
- [ ] Adhesion fields:
    - [ ] Cell A index
    - [ ] Cell B index
    - [ ] Anchor A (local direction vector)
    - [ ] Anchor B (local direction vector)
    - [ ] Rest length
    - [ ] Stiffness
    - [ ] Damping coefficient
    - [ ] Orientation stiffness
    - [ ] Orientation damping
    - [ ] Twist stiffness (optional)
    - [ ] Twist damping (optional)
    - [ ] Twist reference quaternions (optional)
    - [ ] Original side assignment (which was cellA in parent)
    - [ ] Active flag
- [ ] Adhesion count tracker (`usize`)

#### Genome Library

- [ ] Genome deduplication table (`HashMap<GenomeHash, GenomeData>`)
- [ ] Genome reference counting for cleanup
- [ ] Active genome index for current preview
- [ ] Genome data structure:
    - [ ] Mode array `[Mode; 120]`
    - [ ] Genome hash for deduplication
    - [ ] Metadata (name, description)

#### Mode Structure

- [ ] Cell type (`u8` or enum)
- [ ] Child A mode index (`u8`)
- [ ] Child B mode index (`u8`)
- [ ] Split mass (`f32`)
- [ ] Split ratio (`f32`)
- [ ] Split interval (`f32`)
- [ ] Split angle (pitch/yaw) (`Vec2`)
- [ ] Child A angle (pitch/yaw) (`Vec2`)
- [ ] Child B angle (pitch/yaw) (`Vec2`)
- [ ] Color (`RGB` or `Vec3`)
- [ ] Make adhesion flag (`bool`)
- [ ] Child A keep adhesion flag (`bool`)
- [ ] Child B keep adhesion flag (`bool`)
- [ ] Nutrient priority (`f32`)
- [ ] Prioritize when low flag (`bool`)
- [ ] Max connections (`u8`)
- [ ] Min connections (`u8`)
- [ ] Max splits (`i32`, -1 for infinite)
- [ ] Cytoskeleton strength (`f32`)
- [ ] Signal thresholds (TBD - depends on behavior system)

#### Spatial Grid

- [ ] Grid structure (`Vec<Vec<usize>>` - each cell contains list of cell indices)
- [ ] Grid dimensions (`IVec3`)
- [ ] Grid cell size (`f32`)
- [ ] Grid world bounds (`Vec3`)
- [ ] Grid rebuild/update function

### Simulation State

- [ ] Current tick (`u64`)
- [ ] Timestep size (`f32`, default 0.02)
- [ ] Paused flag (`bool`)
- [ ] Target tick (for scrubber) (`u64`)
- [ ] Genome dirty flag (requires simulation reset)
- [ ] Simulation bounds (sphere radius, soft zone parameters)

### Memory Management

- [ ] Cell allocation strategy (swap-and-pop for deletion)
- [ ] Adhesion allocation strategy (swap-and-pop for deletion)
- [ ] Genome cleanup (remove unreferenced genomes)
- [ ] Clear simulation function (reset to initial state)

---

## Physics Systems

### Velocity Verlet Integration

#### Core Integration Loop

- [ ] Calculate forces for all active cells
- [ ] Calculate torques for all active cells
- [ ] Update velocities: `v += (force / mass) * dt`
- [ ] Update angular velocities: `ω += (torque / inertia) * dt`
- [ ] Update positions: `pos += velocity * dt`
- [ ] Update rotations: `rot = integrate_rotation(rot, ω, dt)`
- [ ] Clamp velocities to maximum (stability)
- [ ] Clamp angular velocities to maximum (stability)

#### Integration Helpers

- [ ] Force accumulation per cell (reset each tick)
- [ ] Torque accumulation per cell (reset each tick)
- [ ] Quaternion integration function
- [ ] Inertia calculation (sphere: `I = 0.4 * mass * radius²`)

### Collision Detection

#### Spatial Grid System

- [ ] Grid cell assignment function (position → grid index)
- [ ] Grid rebuild function (clear and repopulate each tick)
- [ ] Neighbor cell lookup (3×3×3 neighborhood)
- [ ] Collision pair generation (only check nearby cells)

#### Collision Response

- [ ] Cell-cell overlap detection
- [ ] Overlap penetration calculation
- [ ] Collision force calculation:
    - [ ] Spring force: `k * overlap * direction`
    - [ ] Damping force: `damping * relative_velocity`
    - [ ] Cytoskeleton strength scaling
- [ ] Handle large penetrations (clamping or non-linear force)
- [ ] Apply equal and opposite forces (Newton's 3rd law)

#### World Boundary Collision

- [ ] Distance from origin calculation
- [ ] Soft zone detection (95-100 units)
- [ ] Soft zone force calculation:
    - [ ] `penetration = (distance - 95) / 5.0`
    - [ ] `force = 500.0 * penetration² * toward_origin`
- [ ] Soft zone torque calculation:
    - [ ] Rotate cells to face inward
    - [ ] `torque = 50.0 * penetration * angle`
- [ ] Hard clamp (beyond 100 units):
    - [ ] Position snap to boundary
    - [ ] Velocity reversal

### Adhesion System

#### Adhesion Creation

- [ ] Create adhesion during cell division (zones A, B, C)
- [ ] Geometric anchor calculation:
    - [ ] Calculate child and neighbor positions in parent frame
    - [ ] Calculate child anchor in child's local frame
    - [ ] Calculate neighbor anchor in neighbor's local frame
- [ ] Adhesion settings inheritance from mode
- [ ] Check adhesion limits (max 10 per cell)
- [ ] Glueocyte contact-based adhesion creation (TBD)

#### Adhesion Forces and Torques

- [ ] Transform anchors to world space (rotate by cell orientation)
- [ ] Linear spring force:
    - [ ] `force = stiffness * (current_distance - rest_length) * direction`
- [ ] Linear damping force:
    - [ ] `damping = -damping_coeff * relative_velocity_along_connection`
- [ ] Orientation spring torque:
    - [ ] Calculate anchor misalignment angle
    - [ ] `torque = axis × angle × orientation_stiffness`
- [ ] Orientation damping torque:
    - [ ] `damping_torque = -axis × angular_velocity_component * damping_coeff`
- [ ] Twist constraint (optional):
    - [ ] Project twist rotation around adhesion axis
    - [ ] `twist_torque = axis × twist_angle × twist_stiffness`
- [ ] Tangential forces:
    - [ ] Convert torques to linear forces
    - [ ] `tangential_force = (torque × position_vector) / distance²`
- [ ] Force/torque accumulation to cells A and B

#### Adhesion Breaking

- [ ] Calculate total force magnitude
- [ ] Compare to breaking threshold
- [ ] Remove adhesion if exceeded
- [ ] Update cell adhesion counts

#### Adhesion Cleanup

- [ ] Remove adhesions referencing deleted cells
- [ ] Compact adhesion array (swap-and-pop)

### Cell Division

#### Division Criteria Check

- [ ] Check mass: `current_mass >= split_mass`
- [ ] Check age: `age >= split_interval`
- [ ] Check adhesions: `min_adhesions <= connections < max_adhesions`
- [ ] Check split count: `splits < max_splits` (or -1)
- [ ] Division pending flag if some criteria met

#### Mass Splitting

- [ ] Calculate child masses:
    - [ ] `child_A_mass = parent_mass * split_ratio`
    - [ ] `child_B_mass = parent_mass * (1.0 - split_ratio)`
- [ ] Parent mass consumed completely

#### Position and Orientation

- [ ] Calculate split direction from split angle (pitch/yaw)
- [ ] Calculate child A position: `parent_pos + split_direction * offset`
- [ ] Calculate child B position: `parent_pos - split_direction * offset`
- [ ] Calculate child A orientation from child A angle
- [ ] Calculate child B orientation from child B angle
- [ ] Apply parent's rotation to child orientations

#### Adhesion Inheritance

- [ ] Classify parent adhesions into zones (A, B, C):
    - [ ] Zone A: anchor opposite to split → Child B
    - [ ] Zone B: anchor same as split → Child A
    - [ ] Zone C: equatorial → Both children
- [ ] For each adhesion:
    - [ ] Calculate inherited anchors geometrically
    - [ ] Create new adhesion(s) for child(ren)
    - [ ] Update neighbor's adhesion to point to child
    - [ ] Preserve original side assignment
- [ ] Handle simultaneous division (both cells dividing)

#### Child Initialization

- [ ] Set child A mode from parent mode's child_A_mode
- [ ] Set child B mode from parent mode's child_B_mode
- [ ] Initialize child ages to 0
- [ ] Initialize child split counts to 0
- [ ] Inherit genome index from parent
- [ ] Set cell types from new modes
- [ ] Initialize velocities (inherit or zero?)
- [ ] Initialize angular velocities (inherit or zero?)

#### Cell Allocation

- [ ] Deterministic allocation (single-threaded, sequential)
- [ ] Child A overwrites parent slot
- [ ] Child B takes next available slot
- [ ] Update cell count
- [ ] Remove parent adhesions
- [ ] Check for cell count limit (256 max)

### Energy and Biomass

#### Mass Growth (Test Cells)

- [ ] Passive nutrient gain: `mass += nutrient_gain_rate * dt`
- [ ] Storage cap enforcement: `mass = min(mass, 2.0 * split_mass)`
- [ ] Radius update: `radius = clamp(mass.min(max_cell_size), 0.5, 2.0)`

#### Mass Consumption (Flagellocyte)

- [ ] Calculate swim force magnitude from signals (TBD)
- [ ] Mass consumption: `mass -= 0.2 * force_magnitude * dt`
- [ ] Check starvation: `if mass < 0.5 { kill cell }`

#### Mass Transfer Between Cells

- [ ] For each adhesion connection:
    - [ ] Calculate pressure A: `mass_A / priority_A`
    - [ ] Calculate pressure B: `mass_B / priority_B`
    - [ ] Calculate flow: `(pressure_A - pressure_B) * transport_rate * dt`
    - [ ] Check minimum thresholds:
        - [ ] Prioritized cells: 0.1 minimum
        - [ ] Others: 0.0 minimum
    - [ ] Transfer mass between cells
- [ ] Emergency priority boost: `10x when mass < 0.6` (if prioritize_when_low enabled)
- [ ] Block transfer during division

#### Cell Death

- [ ] Check starvation: `mass < 0.5`
- [ ] Mark cell as inactive
- [ ] Remove cell from simulation (swap-and-pop)
- [ ] Remove associated adhesions
- [ ] Release nutrients to environment (Preview doesn't have grid - skip for now)
- [ ] Update cell count

---

## Cell Types Implementation

### Cell Type Enum

- [ ] Define CellType enum with all types
- [ ] Implement conversion from u8
- [ ] Default cell type (Test/Chronocyte)

### Passive Cell Types

#### Test Cell (cell_type = 0)

- [ ] Passive mass accumulation (already covered)
- [ ] No special update logic needed

#### Chronocyte

- [ ] Same as Test Cell (splits after time)
- [ ] No additional logic needed

#### Phagocyte

- [ ] Collision detection with food pellets (TBD - no food in Preview yet)
- [ ] Mass transfer from pellet to cell
- [ ] Remove consumed pellet

#### Chemocyte

- [ ] Sample grid cell for dissolved nutrients (TBD - no grid in Preview yet)
- [ ] Mass gain proportional to concentration
- [ ] Deplete local concentration

#### Photocyte

- [ ] Sample grid cell for light intensity (TBD - no grid in Preview yet)
- [ ] Mass gain proportional to light
- [ ] Surface area/orientation effects

#### Lipocyte

- [ ] Higher storage cap: `mass_cap = 3.0 * split_mass` (example)
- [ ] Normal mass transfer participation

#### Nitrocyte

- [ ] Generate nutrients to grid (TBD - no grid in Preview yet)
- [ ] Mass consumption for generation

#### Glueocyte

- [ ] Collision detection with other cells
- [ ] Create adhesion on contact
- [ ] Anchor calculation (predefined or dynamic?)
- [ ] Check adhesion limits

#### Devorocyte

- [ ] Collision detection with other cells
- [ ] Mass drain from contacted cell
- [ ] Transfer rate and mechanism

#### Keratinocyte

- [ ] Armor flag or property
- [ ] Block Devorocyte and Glueocyte effects
- [ ] Implementation depends on their mechanics

### Active Cell Types

#### Flagellocyte

- [ ] Read signal input (TBD - signal system)
- [ ] Calculate propulsion force: `force = signal_strength * max_force * forward_direction`
- [ ] Apply force to cell
- [ ] Mass consumption: `mass -= 0.2 * force_magnitude * dt`

#### Myocyte

- [ ] Read signal inputs (TBD - signal system)
- [ ] Calculate actuation:
    - [ ] Bend in plane 1
    - [ ] Bend in plane 2
    - [ ] Contract/extend
    - [ ] Twist
- [ ] Modify adhesion rest lengths or angles
- [ ] Apply forces/torques
- [ ] Mass consumption

#### Secrocyte

- [ ] Read signal input (TBD - signal system)
- [ ] Calculate secretion rate
- [ ] Add chemicals to grid cell (TBD - no grid in Preview yet)
- [ ] Check saturation limits
- [ ] Mass consumption

#### Stem Cell

- [ ] Read chemical concentrations (TBD - no grid in Preview yet)
- [ ] Determine target mode based on concentrations
- [ ] One-time mode switch
- [ ] Update cell type
- [ ] Continue with new mode rules

#### Neurocyte

- [ ] Read input signals (TBD - signal system)
- [ ] Compute output signals (logic/math)
- [ ] Write output signals
- [ ] Signal computation rules from mode

#### Cilliocyte

- [ ] Read signal input (TBD - signal system)
- [ ] Collision detection with other cells
- [ ] Apply push force to contacted cells
- [ ] Mass consumption

#### Audiocyte

- [ ] Read signal input (TBD - signal system)
- [ ] Generate sound (audio system integration)
- [ ] Mass consumption

### Sensor Cell Types

#### Oculocyte

- [ ] Get forward direction from cell orientation
- [ ] Define detection cone (angle, range from mode)
- [ ] Query spatial grid cells in cone
- [ ] Find targets of specified type (cells, food, wall, etc.)
- [ ] Calculate distance to nearest target
- [ ] Generate signal proportional to distance
- [ ] Write signal to cell's output

#### Senseocyte

- [ ] Query spatial grid cells in radius
- [ ] Find targets of specified type
- [ ] Calculate distance to nearest target
- [ ] Generate omnidirectional signal
- [ ] Write signal to cell's output

#### Stereocyte

- [ ] Query spatial grid for targets
- [ ] Calculate relative position in local coords
- [ ] Generate position vector signal (2D or 3D?)
- [ ] Write signal to cell's output

#### Velocity Sensor

- [ ] Track target from previous tick
- [ ] Calculate relative velocity in local coords
- [ ] Generate velocity vector signal
- [ ] Write signal to cell's output

### Sensor Target System

- [ ] Define target type enum (cells, colors, wall, food, chemicals, light)
- [ ] Target filtering in spatial queries
- [ ] Color matching for cell detection
- [ ] Mode configuration for sensor targets

---

## Genome System Integration

### Genome Editor Connection

- [ ] Register current genome with preview simulation
- [ ] Hot-reload genome when edited
- [ ] Reset simulation to single cell on genome change
- [ ] Validate genome before applying (no circular refs, valid mode indices)

### Genome Change Handling

- [ ] Detect genome modification event
- [ ] Set genome dirty flag
- [ ] Clear current simulation state
- [ ] Spawn initial cell with new genome:
    - [ ] Mode 0 (root mode)
    - [ ] Split-ready mass (or 50% for Test cells)
    - [ ] Center position
    - [ ] Default orientation
    - [ ] Age 0
- [ ] Resume simulation

### Initial Cell Placement

- [ ] Manual placement mode (user clicks to spawn)
- [ ] Default placement (center of world)
- [ ] Initial cell configuration from Mode 0
- [ ] Genome index assignment
- [ ] Add to spatial grid

---

## Rendering System

### Debug Rendering

#### Cell Rendering

- [ ] Icosphere mesh generation (multiple subdivision levels)
- [ ] Instanced rendering for all active cells
- [ ] Instance data buffer:
    - [ ] Position
    - [ ] Rotation (quaternion or matrix)
    - [ ] Radius (scale)
    - [ ] Color from genome mode
- [ ] Update instance buffer each frame
- [ ] Simple phong shading or PBR

#### Adhesion Rendering

- [ ] Line rendering (GL_LINES or mesh)
- [ ] For each active adhesion:
    - [ ] Get cell A and B positions
    - [ ] Draw line from A to B
    - [ ] Color: white or based on strain
    - [ ] Thickness: fixed or based on strength

#### Gizmos and Debug Info

- [ ] Selected cell highlight (outline or color)
- [ ] Orientation gizmo (XYZ axes)
- [ ] Adhesion anchor gizmos (small spheres at anchor points)
- [ ] Split plane gizmo (disk showing split plane)
- [ ] Velocity vectors (optional)
- [ ] Force vectors (optional)

### Wireframe Mode

- [ ] Toggle wireframe rendering
- [ ] Wireframe + solid overlay option
- [ ] Shader support for wireframe on icospheres

### Cell Selection

- [ ] Mouse raycast into scene
- [ ] Ray-sphere intersection test with all cells
- [ ] Find nearest intersected cell
- [ ] Highlight selected cell
- [ ] Display cell info in UI

### World Boundary Rendering

- [ ] Semi-transparent sphere at radius 100
- [ ] Icosphere mesh (subdivision level 7)
- [ ] Fresnel edge lighting effect
- [ ] Configurable opacity
- [ ] Soft zone visualization (optional gradient at 95-100)

### Camera Integration

- [ ] Use existing 6DOF camera system
- [ ] Use existing orbital camera system
- [ ] Smooth camera transitions
- [ ] Focus on selected cell option

### Performance Rendering Notes

- [ ] With 256 cells max, LOD probably not critical
- [ ] Frustum culling still beneficial
- [ ] No need for occlusion culling at this scale
- [ ] Keep rendering simple for preview

---

## UI Integration

### Time Scrubber

- [ ] Slider UI element (tick 0 to current max tick)
- [ ] Seek to specific tick
- [ ] Pause simulation on scrub
- [ ] Resume simulation option
- [ ] Display current tick number
- [ ] Display time in seconds (tick * dt)

### Pause/Play Controls

- [ ] Pause button
- [ ] Play button
- [ ] Step forward one tick button
- [ ] Step backward one tick button (requires state saving)
- [ ] Reset to tick 0 button

### Speed Controls

- [ ] Not needed for Preview (manual scrubber only)
- [ ] Preview runs as fast as possible when playing

### Simulation Info Display

- [ ] Current tick
- [ ] Current time (seconds)
- [ ] Cell count (current / max 256)
- [ ] Adhesion count
- [ ] FPS (frames per second)
- [ ] TPS (ticks per second)
- [ ] Performance: Frieza units per cell
- [ ] Physics step time
- [ ] Rendering time

### Selected Cell Info Panel

- [ ] Cell index
- [ ] Mode name and index
- [ ] Cell type
- [ ] Position (x, y, z)
- [ ] Velocity (x, y, z)
- [ ] Rotation (quaternion or euler)
- [ ] Angular velocity
- [ ] Mass / biomass
- [ ] Radius
- [ ] Age (ticks and seconds)
- [ ] Split count
- [ ] Adhesion count
- [ ] Genome index
- [ ] Nutrient priority
- [ ] Signal values (inputs/outputs)

### Real-time Update on Genome Change

- [ ] Listen for genome editor events
- [ ] Automatically reset simulation
- [ ] Maintain pause/play state
- [ ] Show "genome updated" notification

### Genome Editor Integration

- [ ] Preview window embedded in editor, or separate
- [ ] Sync genome reference between editor and preview
- [ ] Visual feedback when genome invalid

---

## Performance Optimization

### Single-Threaded Optimization

- [ ] Cache-friendly data layout (SoA)
- [ ] Minimize cache misses
- [ ] Avoid unnecessary allocations
- [ ] Reuse buffers across ticks

### Spatial Grid Optimization

- [ ] Efficient grid cell indexing
- [ ] Minimize grid cell count (balance size vs granularity)
- [ ] Fast neighbor lookup (pre-computed offsets)
- [ ] Only rebuild when necessary (every tick for now)

### Collision Detection Optimization

- [ ] Early exit for distant cells
- [ ] Broad phase via spatial grid
- [ ] Narrow phase only for nearby pairs
- [ ] SIMD for distance calculations (optional)

### Adhesion Optimization

- [ ] Process only active adhesions
- [ ] Compact adhesion array (no gaps)
- [ ] Cache anchor transformations

### Profiling Points

- [ ] Grid rebuild time
- [ ] Collision detection time
- [ ] Adhesion force calculation time
- [ ] Integration time
- [ ] Cell division time
- [ ] Total physics step time
- [ ] Rendering time
- [ ] Overall tick time

### Performance Target Validation

- [ ] Measure performance with 256 cells
- [ ] Ensure < 2Fr (< 0.512ms per cell per tick)
- [ ] 256 cells × 30 ticks in 16ms = 0.53ms per tick
- [ ] Profile and optimize bottlenecks

### Memory Usage

- [ ] Fixed-size arrays (no dynamic growth)
- [ ] Minimal heap allocations per tick
- [ ] Genome data shared, not duplicated
- [ ] Memory footprint: estimate and measure

---

## Testing & Validation

### Unit Tests

#### Physics Tests

- [ ] Verlet integration accuracy (energy conservation)
- [ ] Collision detection (overlap calculation)
- [ ] Collision response (force magnitude, direction)
- [ ] World boundary forces (soft zone, hard clamp)
- [ ] Adhesion force calculations (linear, angular, tangential)
- [ ] Cell division (mass splitting, position calculation)
- [ ] Adhesion inheritance (zone classification, anchor calculation)

#### Genome Tests

- [ ] Mode indexing (valid child references)
- [ ] Genome deduplication (hash collisions, cleanup)
- [ ] Genome validation (circular references, orphaned modes)
- [ ] Initial cell creation from Mode 0

#### Data Structure Tests

- [ ] Cell allocation (swap-and-pop correctness)
- [ ] Adhesion allocation (swap-and-pop correctness)
- [ ] Spatial grid (correct cell binning, neighbor queries)
- [ ] Cell count tracking (increments, decrements)

### Integration Tests

#### Simulation Tests

- [ ] Run 30 ticks without crashes
- [ ] Energy conservation (total energy constant or decreasing)
- [ ] Momentum conservation (linear, angular)
- [ ] Cell division produces correct cell count
- [ ] Adhesions maintained after division
- [ ] Genome hot-reload resets simulation correctly

#### Organism Development Tests

- [ ] Single cell → organism growth
- [ ] Symmetric division (split_ratio = 0.5)
- [ ] Asymmetric division (split_ratio != 0.5)
- [ ] Chain organism (linear adhesions)
- [ ] Branching organism (tree structure)
- [ ] Complex organism (multiple cell types)

### Determinism Tests

- [ ] Run same genome twice, identical results
- [ ] Bit-for-bit identical state at tick N
- [ ] Identical cell positions across runs
- [ ] Identical adhesion configurations across runs
- [ ] Save state at tick N, reload, continue → same result

### Performance Tests

- [ ] Measure tick time with 1, 10, 100, 256 cells
- [ ] Verify < 2Fr at 256 cells
- [ ] Measure grid rebuild time
- [ ] Measure collision detection time
- [ ] Measure adhesion calculation time
- [ ] Identify bottlenecks

### Stress Tests

- [ ] 256 cells all dividing simultaneously
- [ ] 256 cells with maximum adhesions each
- [ ] 256 cells clustered in small area (collision stress)
- [ ] 256 cells spread across world (grid stress)
- [ ] Rapid genome changes (hot-reload stress)

### Visual Tests

- [ ] Organisms develop as expected
- [ ] Cell colors match genome
- [ ] Adhesions render correctly
- [ ] Gizmos display correctly
- [ ] Selection highlighting works
- [ ] Wireframe mode works
- [ ] World boundary renders correctly

### Edge Cases

- [ ] Cell at exactly world boundary (100 units)
- [ ] Cell moving fast toward boundary
- [ ] Cell division at cell count limit (256)
- [ ] Cell division with max adhesions (10)
- [ ] Adhesion breaking under extreme force
- [ ] Zero mass cell (starvation edge case)
- [ ] Negative mass prevention
- [ ] Division with split_ratio = 0 or 1 (edge cases)
- [ ] Invalid genome reference (missing mode)

---

## Implementation Phases

### Phase 1: Foundation

- [ ] Set up Preview module structure
- [ ] Implement core data structures (cells, adhesions, genome)
- [ ] Implement spatial grid
- [ ] Basic cell allocation (add, remove)
- [ ] Simple rendering (debug icospheres)

### Phase 2: Basic Physics

- [ ] Velocity Verlet integration
- [ ] Simple collision detection (no grid yet)
- [ ] Collision response forces
- [ ] World boundary forces
- [ ] Test with a few cells

### Phase 3: Adhesions

- [ ] Adhesion data structure
- [ ] Adhesion force calculations (linear spring)
- [ ] Adhesion rendering (lines)
- [ ] Test with manually created adhesions

### Phase 4: Cell Division

- [ ] Division criteria checking
- [ ] Mass splitting
- [ ] Position/orientation calculation
- [ ] Adhesion inheritance (zones A, B, C)
- [ ] Geometric anchor recalculation
- [ ] Test with simple organisms (linear chains)

### Phase 5: Genome Integration

- [ ] Connect to genome editor
- [ ] Spawn initial cell from Mode 0
- [ ] Hot-reload on genome change
- [ ] Validate genome structure
- [ ] Test with various genome configurations

### Phase 6: Energy System

- [ ] Mass accumulation (Test cells)
- [ ] Mass consumption (Flagellocytes)
- [ ] Mass transfer between cells (priority system)
- [ ] Cell death and removal
- [ ] Test energy conservation

### Phase 7: Cell Types - Passive

- [ ] Implement CellType enum
- [ ] Chronocyte (basic)
- [ ] Lipocyte (higher storage)
- [ ] Defer grid-dependent types (Phagocyte, Chemocyte, Photocyte, Nitrocyte)
- [ ] Test multi-type organisms

### Phase 8: UI Integration

- [ ] Time scrubber control
- [ ] Pause/play controls
- [ ] Info displays (tick, cell count, performance)
- [ ] Cell selection (raycasting)
- [ ] Selected cell info panel
- [ ] Test all UI interactions

### Phase 9: Polish and Optimization

- [ ] Profile and optimize bottlenecks
- [ ] Achieve < 2Fr target
- [ ] Improve rendering quality
- [ ] Add gizmos and visual aids
- [ ] Improve camera controls
- [ ] Bug fixes

### Phase 10: Advanced Features

- [ ] Glueocyte (contact adhesions)
- [ ] Devorocyte (mass drain)
- [ ] Keratinocyte (armor)
- [ ] Signal system foundation (for active cells)
- [ ] Defer full sensor/signal implementation to later

---

## Future Extensions (Post-MVP)

### Grid-Based Features

Once fluid grid is implemented, add:

- [ ] Phagocyte (food pellet consumption)
- [ ] Chemocyte (dissolved nutrient absorption)
- [ ] Photocyte (light-based energy)
- [ ] Nitrocyte (nutrient generation)
- [ ] Secrocyte (chemical secretion)
- [ ] Chemical diffusion preview

### Signal and Behavior System

- [ ] Define signal data structures
- [ ] Signal propagation (distance-based)
- [ ] Neurocyte signal computation
- [ ] Flagellocyte propulsion
- [ ] Myocyte actuation
- [ ] Cilliocyte pushing
- [ ] Sensor cells (Oculocyte, Senseocyte, Stereocyte, Velocity)

### Advanced Rendering

- [ ] Smooth cell deformation (metaballs/marching cubes)
- [ ] Better materials and lighting
- [ ] Shadow casting
- [ ] Post-processing effects
- [ ] Animation and effects (division, death)

### State Saving

- [ ] Save preview state at any tick
- [ ] Load saved states
- [ ] Compare states across ticks
- [ ] State history for scrubbing (requires state snapshots)

---

## Known Limitations (Preview Mode)

- **No fluid simulation:** Preview doesn't include the 64³ grid
- **No environmental systems:** No chemicals, food pellets, light grid
- **Limited cell types:** Grid-dependent types deferred
- **No backwards scrubbing:** Would require state snapshots
- **Manual time control only:** No automatic playback speed
- **256 cell hard limit:** No dynamic expansion

---

## Success Criteria

### Functional Requirements Met

- [ ] 256 cells simulated deterministically
- [ ] Velocity Verlet integration at 50 TPS
- [ ] Cell division with adhesion inheritance
- [ ] Genome hot-reload and validation
- [ ] Time scrubber control
- [ ] Real-time organism preview
- [ ] Cell selection and info display

### Performance Requirements Met

- [ ] < 2Fr (256 cells × 30 ticks in 16ms)
- [ ] Smooth rendering at 60 FPS
- [ ] No frame drops during division
- [ ] Instant genome reload (< 50ms)

### Quality Requirements Met

- [ ] Bit-for-bit deterministic (same genome = same result)
- [ ] Energy conservation (no mass creation)
- [ ] Momentum conservation (linear and angular)
- [ ] Stable long-term simulation (no explosions)
- [ ] Accurate organism development (matches genome)

---

## Notes

### Design Decisions

- Single-threaded for perfect determinism
- Fixed-size arrays for predictable performance
- SoA layout for cache efficiency
- Swap-and-pop deletion for stable indices within a tick
- Geometric anchor recalculation for drift prevention
- Zone-based adhesion inheritance for correctness

### Common Pitfalls to Avoid

- Don't forget Newton's 3rd law (equal/opposite forces)
- Don't let adhesions reference deleted cells
- Don't create race conditions (single-threaded helps)
- Don't skip validation (genomes can be invalid)
- Don't ignore performance (profile early, profile often)
- Don't accumulate floating-point error (use good integration)

### Reference Implementations

- C++ version (match behavior exactly)
- Cell Lab mechanics (inspiration for cell types)
- Existing adhesion force calculations (documented)

---

## Changelog

**v1.0** - Initial comprehensive checklist

- All core systems documented
- Implementation phases planned
- Success criteria defined