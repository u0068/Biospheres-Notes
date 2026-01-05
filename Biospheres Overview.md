# Biospheres - Comprehensive Design Document

## Overview

Biospheres is a 3D biological simulation game inspired by Cell Lab, featuring massive cellular ecosystems with 100K-200K cells. The simulation uses Rust, Bevy, WebGPU, and WGSL compute shaders across three distinct modes: Preview (256 cells), CPU (5K-10K cells), and GPU (100K-200K cells).

**Performance Note:** `1 Frieza (Fr) = 1 microsecond per cell per tick`

---

## Glossary

### Core Concepts

- [[#Simulation Modes]] - Preview, CPU, and GPU simulation configurations
- [[#Performance Targets]] - TPS and Fr benchmarks for each mode
- [[#Stability Requirements]] - Determinism, energy conservation, momentum
- [[#World Boundaries]] - Spherical boundary system with soft constraints

### Physics Systems

- [[#Integration Method]] - Velocity Verlet at 50 TPS
- [[#Cell Collisions]] - Soft collision system with spatial partitioning
- [[#Cell Adhesions]] - Spring-damper force system with anchors
- [[#Spatial Partitioning]] - Uniform grid for collision detection
- [[#Fluid Simulation]] - 64³ grid-based fluid dynamics
- [[#Two-Way Coupling]] - Cell-fluid interaction system

### Cell Biology

- [[#Cell Types]] - All available cell specializations
- [[#Genome System]] - Mode-based differentiation system
- [[#Cell Division]] - Deterministic allocation and splitting
- [[#Energy/Biomass Economy]] - Mass-based resource system
- [[#Cell Signaling]] - 4-substance chemical communication
- [[#Sensor Systems]] - Detection and signal generation
- [[#Behavior System]] - Signal-driven cell actions ⚠️

### Environmental Systems

- [[#Chemical Diffusion]] - Grid-based chemical transport ⚠️
- [[#Food System]] - Pellet spawning and consumption ⚠️
- [[#Light System]] - Grid-based light field with occlusion ⚠️
- [[#Dissolved Nutrients]] - Chemocyte food source
- [[#Gas System]] - Dissolved gases in fluid

### UI/UX Systems

- [[Camera]] - 6DOF and orbital camera modes
- [[#Genome Editor]] - Visual editor with node graph
- [[#Tool System]] - Cell interaction tools ⚠️
- [[#Genome Hotbar]] - Quick-access saved genomes ⚠️
- [[#Debug Information]] - Performance and world info

### Rendering

- [[#Rendering Pipeline]] - LOD, culling, and optimization ⚠️
- [[#Cell Rendering]] - Mesh-based and impostor approaches
- [[#Adhesion Rendering]] - Line visualization
- [[#Debug Rendering]] - Wireframe and gizmos

### Data Management

- [[#Genome Storage]] - Deduplication and memory layout
- [[#Save/Load System]] - Binary saves and genome export
- [[#Mutation System]] - Genome variation ⚠️

**Legend:** ⚠️ = Needs Further Review

---

## Simulation Modes

### [[Preview Simulation]]

**Purpose:** Fast, deterministic iteration during genome editing

- **Cell Count:** 256 cells maximum
- **Performance Target:** `2Fr = 256 cells × 30 ticks in 16ms`
- **Execution:** Single-threaded for perfect determinism
- **Control:** Manual time scrubber only
- **Updates:** Real-time on genome changes

### [[GPU Simulation]]

**Purpose:** Massive-scale real-time simulation

- **Cell Count:** 100K-200K cells
- **Performance Target:** `100K cells at >50tps = <0.2Fr`
- **Execution:** Compute shader-driven
- **Control:** Variable speed controls, no backwards simulation
- **Features:** Full fluid and gas simulation
- **Requirement:** Same results as CPU and Preview (within acceptable tolerance)

---

## Stability Requirements

### Determinism

- ✅ **Preview Mode:** Bit-for-bit deterministic (single-threaded)
- ⚠️ **CPU/GPU Modes:** As close as possible, focused on long-term stability
    - Multi-hour or multi-day runs without degeneration
    - Acceptable variance between modes, but no catastrophic divergence

### Energy and Momentum Conservation

- **Energy:** Constant or lost (no creation)
- **Linear Momentum:** Conserved in closed system
- **Angular Momentum:** Conserved in closed system
- **Stability Strategy:**
    - Force damping to improve stability
    - Hard constraints on maximum velocities/forces
    - ❌ **No corrective normalization** (would break determinism)

### Integration Method

**Velocity Verlet at 50 TPS (0.02s timestep)**

- **Choice Rationale:** Better energy conservation than Euler
- **Fixed Timestep:** 0.02s (arbitrary, makes math easier)
- **No Adaptive Stepping:** Maintains determinism
- **Physics-Rendering Decoupling:** Prioritize ticks over frames down to 1 FPS

---

## Performance Targets

### Optimization Strategies

#### Physics Optimizations

- **Data-Oriented Design:** Structure of Arrays (SoA)
- **Spatial Partitioning:** Uniform grid (easier than octrees for this use case)
    - Octrees better if world >> cell count, but uniform grid sufficient
    - Used for: collisions, fluid coupling, selection raycasting

#### Graphics Optimizations

- Framerate limiting (don't render more than display refresh)
- Instanced rendering with compact instance data
- Backface culling
- Frustum culling
- Occlusion culling
- Cluster/coarse culling
- GPU-driven LOD
- Front-to-back rendering / Depth pre-pass
- Sphere impostors instead of full geometry
    - Spherical billboards
    - Point sprites
    - Raytracing/raymarching (best looking, simple intersection tests)

---

## World Boundaries

### Spherical Boundary System

**Two-Layer Approach:**

#### Soft Zone (95-100 units from origin)

Primary boundary mechanism - gentle force field:

```
penetration = (distance - 95) / 5.0  // 0.0 to 1.0
force = 500.0 × penetration²         // Quadratic curve
direction = toward origin
```

- **Force Characteristics:**
    - Starts gentle at 95 units
    - Becomes strong at 99 units
    - Maximum 500 units at boundary
- **Rotational Correction:**
    - Torque to rotate cells inward
    - Strength: `50.0 × penetration × angle`

#### Hard Clamp (beyond 100 units)

Safety mechanism for boundary violations:

```
if distance > 100.0:
    position = direction × 100.0        // Snap to surface
    velocity -= radial_velocity × 2.0   // Reverse outward motion
```

### World Parameters

- **Sphere Radius:** 100 units
- **Spatial Grid:** 200×200×200 cubic volume
- **Soft Zone Start:** 95 units
- **Soft Zone Thickness:** 5 units

### Visual Rendering

- Semi-transparent sphere mesh
- Icosphere with 7 subdivisions
- Fresnel edge lighting (glows at edges)
- Configurable opacity and emissive intensity

---

## Cell Collisions

### Soft Collision System

- **Force Model:** Proportional to overlap
- **Cytoskeleton Gene:** Affects collision stiffness
- **Spatial Optimization:** Grid-based collision detection

### ⚠️ Large Penetration Handling - NEEDS REVIEW

**Problem:** Deep overlaps can cause instability with spring-based repulsion

**Potential Solutions:**

- Clamp maximum repulsion force per collision <-- Best solution imo -h
- Limit maximum overlap distance in calculation
- Non-linear spring force (e.g., `force = k × tanh(overlap)`)
- Other approaches TBD

**Testing Required:** Determine best approach for long-run stability

---

## Cell Division

### Division Criteria

Cell must meet ALL conditions:

- ✅ `current_mass >= split_mass`
- ✅ `age >= split_interval`
- ✅ `min_adhesions <= connections < max_adhesions`
- ✅ `splits < max_splits` (or -1 for infinite)

### Deferred Division

- Cells meeting some but not all criteria enter "division pending" state
- Division deferred until ALL conditions met
- Cell continues normal function while waiting

### Mass Splitting

```
child_A_mass = parent_mass × split_ratio
child_B_mass = parent_mass × (1.0 - split_ratio)
```

- Parent mass completely consumed (no loss/creation)
- `split_ratio` determines symmetry (0.5 = symmetric)

### Deterministic Allocation

- ✅ **GPU:** Prefix-sum and scatter allocator
- ✅ **CPU:** Single-threaded compaction
- **Critical:** No race conditions, reproducible results

---

## Cell Adhesions

### Anchor System

**Genome-Defined Anchors (Not Spatial)**

- Anchors stored as normalized direction vectors in local space
- Rotate with cell orientation
- Produces repeatable organism morphology regardless of motion during development

### Anchor Count and Limits

- **Maximum per cell:** 10 adhesions
- **Mode-defined:** Min/max adhesions required before splitting
- **Overflow handling:** Adhesions beyond limit are ignored and don't form

### Inheritance During Division

#### Zone-Based Classification

Adhesions classified by anchor direction relative to split:

- **Zone A:** Opposite to split → inherited by Child B
- **Zone B:** Same as split → inherited by Child A
- **Zone C:** Equatorial band → inherited by BOTH children

#### Geometric Recalculation

Anchors geometrically recalculated (not copied):

1. Calculate child and neighbor positions in parent frame
2. Child anchor: direction from child to neighbor → transform to child's local frame
3. Neighbor anchor: direction from neighbor to child → transform to neighbor's local frame

**Benefits:**

- Zone C neighbors get two separate anchors (one per child)
- Prevents drift from physics perturbations
- Fixed radius (1.0) prevents growth effects

### Force Components

#### 1. Linear Spring Force (Position)

```
force = stiffness × (current_distance - rest_length) × direction
```

- Pulls cells together if too far
- Pushes apart if too close
- Equal and opposite (Newton's 3rd law)

#### 2. Linear Damping Force

```
damping = -damping_coefficient × relative_velocity_along_connection
```

- Opposes relative motion
- Prevents oscillation
- Applied along connection axis

#### 3. Orientation Spring Torque

```
torque = axis × angle × orientation_stiffness
```

- Aligns each cell's anchor with adhesion direction
- Axis perpendicular to both anchor and adhesion
- Maintains relative orientation between cells

#### 4. Orientation Damping Torque

```
damping_torque = -axis × angular_velocity_component × damping_coefficient
```

- Opposes angular motion around correction axis
- Prevents rotational oscillation

#### 5. Twist Constraint (Optional, Disabled by Default)

```
twist_torque = adhesion_axis × twist_angle × twist_stiffness
```

- Prevents twisting around adhesion axis
- Strong damping on relative angular velocity
- Can make anchors appear to "follow" unnaturally

#### 6. Tangential Forces (Shape Maintenance)

```
tangential_force = (total_torque × position_vector) / distance²
```

- Converts rotational corrections to linear motion
- Equal and opposite (conserves momentum)
- Prevents phantom drift from unbalanced forces

### Physics Properties

- Linear springs: Hooke's law with damping
- Angular springs: Orientation alignment torques
- Angular constraints: Hard stops (optional)
- Break threshold: Max force before breaking (prevents instability)

---

## Cell Types

### Passive Cells (No Signal-Based Functions)

#### Chronocyte

- Splits after set time
- No special resource requirements

#### Phagocyte

- Eats food pellets to gain biomass
- Collision or proximity-based consumption (TBD)

#### Chemocyte

- Absorbs dissolved nutrients from environment
- Rate based on concentration gradient (TBD)

#### Photocyte

- Absorbs light to gain biomass
- Rate based on light intensity (TBD)

#### Lipocyte

- Stores extra nutrients as fat
- Provides buffer for organism

#### Nitrocyte

- Generates nitrates
- Adds nutrients to environment

#### Glueocyte

- Creates adhesions with cells it contacts
- Uses predefined or dynamic anchors? (TBD)
- Counts against 10-adhesion limit

#### Devorocyte

- Drains biomass from contacted cells
- Transfer rate (TBD)

#### Keratinocyte

- Protects from Devorocytes and Glueocytes
- Armor mechanism (TBD)

### Active Cells (Signal-Driven Functions)

#### Flagellocyte

- Propels itself forward
- Consumes mass: `0.2 mass/second at full force`
- Force proportional to signal strength

#### Myocyte

- Bends in 2 planes
- Contracts/extends
- Twists
- **⚠️ NEEDS REVIEW:** Multiple simultaneous actuations vs sequential modes?

#### Secrocyte

- Secretes chemicals into environment
- Output rate and biomass cost (TBD)
- Can saturate grid cells

#### Stem Cell

- Differentiates into another mode based on chemical concentrations
- One-time transformation
- Follows coded rules of new mode

#### Neurocyte

- Generates signals based on other signals
- ⚠️ **NEEDS REVIEW:** Logic gates, math functions, or more complex?

#### Cilliocyte

- Pushes cells it contacts
- Force magnitude (TBD)

#### Audiocyte

- Makes sounds based on signals
- Audio system integration (TBD)

### Sensor Cells (Generate Signals)

#### Oculocyte

- Senses presence/distance in forward direction
- Cone-based detection checking grid cells
- Signal proportional to distance
- Range configurable per mode

#### Senseocyte

- Senses presence/distance from target (omnidirectional)
- May merge with Oculocyte due to similarity
- Signal proportional to distance

#### Stereocyte

- Outputs signal proportional to relative position (local coords)
- Detection radius (TBD)

#### Velocity Sensor

- Outputs signal proportional to relative velocity (local coords)
- Target tracking (TBD)

### Sensor Detection Targets

Genome-defined entity types:

- Other cells
- Specific colors
- World boundary
- Food pellets
- Chemical concentrations
- Light sources

### ⚠️ Sensor System - NEEDS FURTHER REVIEW

**Questions:**

- Maximum detection range per sensor type
- Occlusion through other cells
- Stereocyte/Velocity sensor radius requirements
- Performance optimization for 100K cells running sensors (spatial grid queries)
- Filter specificity (detect any cell vs specific cell types)

---

## Cell Signaling

### Signal Substances

- **Count:** 4 signaling substances (like Cell Lab)
- **Value Range:** -1.0 to 1.0 (bipolar for more behavioral complexity)
- **Propagation:** Distance-based falloff over set distances
- **⚠️ NEEDS REVIEW:** Exact falloff curves, propagation speed, diffusion vs instant

### Signal Usage

- **Sensors → Signals:** Generate based on stimuli
- **Signals → Actions:** Drive active cell behaviors
- **Neurocytes:** Transform/combine signals

---

## Behavior System

### ⚠️ NEEDS COMPREHENSIVE REVIEW

**Critical Design Questions:**

#### Control Architecture

How do cells decide actions?

- Simple threshold triggers? (if signal_A > 0.5 → activate)
- Continuous response curves? (output ∝ input)
- Genome-defined behavior graphs/state machines?
- Neural network weights stored in mode?

#### Neurocyte Implementation

"Generates signals based on other signals":

- Basic logic gates (AND/OR/NOT)?
- Mathematical functions (multiply, add, integrate over time)?
- More complex computation?

#### Myocyte Actuation

"Bends in 2 planes, contracts/extends, twists":

- Three separate modes to switch between?
- Simultaneous actuation with varying intensities?
- How are angles/amounts encoded (fixed per mode or signal-controlled)?

#### Flagellocyte Response

- Force proportional to signal strength?
- On/off threshold triggering?
- Directional control mechanism?

#### General Behavior Pattern

- Genome defines behavior rules per mode
- Signals as input → behavior mapping → physical actions
- Needs complete specification before implementation

---

## Energy/Biomass Economy

### Universal Currency: Mass

Mass serves as both physical mass and cellular energy

### Mass Accumulation

#### Test Cells (cell_type == 0)

- Passive growth via `nutrient_gain_rate`
- Storage cap: `2 × split_mass`
- Visual growth: `radius = mass.min(max_cell_size).clamp(0.5, 2.0)`

#### Flagellocyte Cells (cell_type == 1)

- Consumption: `0.2 mass/second at full force`
- Must balance intake vs swimming cost
- Can die from starvation

### Mass Transfer Between Cells

#### Pressure-Based Equilibrium

```
pressure = mass / nutrient_priority
flow = (pressure_A - pressure_B) × transport_rate × dt
Equilibrium: mass_A / mass_B = priority_A / priority_B
```

#### Priority System

- **Base priority:** Set per mode (default 1.0)
- **Emergency boost:** 10× when mass < 0.6
- **Priority flag:** `prioritize_when_low` enables emergency boost
- Higher priority attracts more nutrients

#### Flow Rules

- Flow from high to low pressure
- Minimum thresholds prevent depletion:
    - Prioritized cells: 0.1 minimum
    - Others: 0.0 minimum
- Blocked during cell division

### Division Economics

- Each child starts with allocated mass
- Must grow to own `split_mass` threshold
- Initial conditions:
    - Test cells: 50% of split_mass
    - Other types: 100% of split_mass

### Death and Starvation

- **Death threshold:** mass < 0.5
- Causes:
    - Excessive Flagellocyte swimming
    - Drainage by high-priority neighbors
- **Removal:** Swap-and-pop (efficient deletion)

### ⚠️ Resource Intake - NEEDS SPECIFICATION

**Phagocyte (food pellets):**

- Instant transfer or digestion over time?
- Collision vs proximity-based?
- Whole pellet or gradual consumption?

**Chemocyte (dissolved nutrients):**

- Rate based on surface area?
- Concentration gradient?
- Fixed rate?

**Photocyte (light):**

- Rate based on light intensity?
- Surface area exposure?
- Cell orientation matters?

**Metabolic Costs:**

- Do all cells have upkeep cost?
- Only active cells consume?
- Myocyte actuation cost?
- Secrocyte secretion cost?

**Biomass Transfer:**

- Can cells share biomass through adhesions? (YES - already specified)
- Circulatory system analog via priority system? (YES - already specified)
- Individual cell independence? (NO - connected cells share)

---

## Genome System

### Mode Structure

**Mode = Complete Cell Differentiation State**

Each mode contains:

- Cell type selection
- Child A mode reference
- Child B mode reference
- Split mass threshold
- Split ratio (asymmetric division)
- Split angle
- Child A angle
- Child B angle
- Color information (RGB)
- Make adhesion flag
- Child A keep adhesion flag
- Child B keep adhesion flag
- Nutrient priority
- Split interval (time)
- Max splits (-1 for infinite)
- Min/max adhesions required for split
- Signal thresholds and behaviors (TBD)
- ~15-20+ parameters total per mode

### Mode Count

- **Fixed array:** 120 modes (configurable default)
- **Memory layout:** GPU can trivially calculate position
- **Indexing:** Modes 0-119, direct array access
- **Unused modes:** Simply empty, no placeholder data
- **Naming:** User can rename modes at will

### Genome Storage

- **Size:** ~200 bytes per mode × 120 = 24KB per unique genome
- **Target:** 40KB total with additional future parameters
- **Deduplication:**
    - Most organisms share identical genomes
    - Stored once in deduplication table
    - Cells reference by pointer/index
- **Optimization:**
    - Self-referential mode compression
    - Identical mode merging

### Genome Editing Workflow

1. **Preview Mode Only:** All genome editing restricted to Preview
2. **Validation:** Check for circular references, orphaned modes
3. **Injection:** Transfer edited genome to CPU/GPU sims
    - ⚠️ Injection mechanism (TBD): spawn new cell vs replace existing
    - ⚠️ Pause required or inject while running?
    - ⚠️ Reset to single cell or preserve development state?

### Human-Readable Format

- Text-based genome files
- **Delta encoding:** Only show modes changed from defaults
- **Readability:** Easy to share and version control
- **Upgrade utility:** Warns about breaking changes, migrates format

---

## Mutation System

### ⚠️ NEEDS COMPREHENSIVE REVIEW

**Critical Design Questions:**

#### Mutation Timing

- During cell division only?
- Random chance over time?
- User-triggered?
- Environmental triggers (radiation zones)?
- Per-division mutation rate vs time-based probability?

#### Mutation Scope

What can mutate:

- Mode parameters (split mass, angles, colors, priorities)?
- Cell types (Photocyte → Flagellocyte)?
- Child mode references (rewiring developmental tree)?
- All parameters or restricted subset?

#### Mutation Rate Control

- Per-organism genome setting?
- Global simulation setting?
- Both with override capability?
- Can mutations be disabled entirely in some modes?
- Different rates for different parameter types?

#### Genome Deduplication with Mutations

- Does each mutation create new unique genome entry?
- When are mutated genomes garbage collected (organism death)?
- Memory management for thousands of unique mutated genomes
- Mutation tracking/lineage system?

#### Mutation Effects

- Pure random parameter tweaks within valid ranges?
- Beneficial/neutral/harmful classification?
- No fitness evaluation (sandbox)?
- Can mutations be lethal (invalid genomes fail to develop)?
- Mutation magnitude (small tweaks vs large jumps)?

#### Selection and Evolution

- Is there selection pressure driving evolution?
- User-defined fitness criteria?
- Purely sandbox random mutations?
- Survival of the fittest mechanics?
- Population dynamics and speciation?

---

## Spatial Partitioning

### Uniform Grid Implementation

- **Structure:** Uniform grid over world volume
- **Dimensions:** 200×200×200 cells covering sphere
- **Usage:** Cell collisions, fluid coupling, selection raycasting
- **Shared:** Identical structure for GPU and CPU sims

### Update Strategy

- Rebuild every tick or incremental updates? (TBD)
- With 200K cells moving, rebuild cost critical
- GPU parallel build vs CPU sequential build

### Grid Cell Size

- Cell radius ~1.0 unit
- Grid cell size comparable to cell size
- Multiple cells per grid cell expected
- Spatial queries return cell lists per grid cell

---

## Fluid Simulation

### Grid Structure

- **Resolution:** 64³ = ~262,000 grid cells
- **Coverage:** Entire spherical world boundary
- **Cell Size:** ~3.125 units (200 / 64)
- **Relationship:** Cells are roughly same size or smaller than grid cells

### Per-Grid-Cell Data

- Fluid velocity (3D vector)
- Dissolved nutrients (type count TBD)
- 4 chemical signal substances
- Light intensity (scalar or directional TBD)
- Temperature (for convection TBD)
- Gas concentrations (O2, CO2, etc. TBD)
- Other substances (TBD)

### Update Strategy

- ⚠️ **Adaptive updates:** Cells with active transfers update more frequently
- **Static cells:** Less frequent updates
- **Empty cells:** Minimal updates
- **Challenge:** Tracking active cells and scheduling on GPU
- **Update frequency:** Every tick (50 TPS) or less? (TBD)

### Simulation Method

- Full Navier-Stokes or simplified advection-diffusion? (TBD)
- Diffusion solver: explicit, implicit, or semi-implicit? (TBD)
- Pressure solver required for incompressible flow? (TBD)

---

## Two-Way Coupling

### Cell → Fluid

- **Blocking:** Cells act as obstacles redirecting flow
- **Perturbations:** Propelling cells (Flagellocytes) create currents
- **Organism Groups:** Treated as collection of individual cells, not merged

### Fluid → Cell

- **Drag forces:** Cells pushed by fluid currents
- **Force application:** Based on velocity differential
- **Drag coefficient:** Per cell type or universal? (TBD)

### Binning Strategy

- Cells write forces/velocities into grid cells they occupy
- **Atomic operations** required on GPU for multiple cells per grid cell
- **Single cell per grid:** Cells only affect grid cell containing center
- **Multi-cell influence:** Larger cells affect neighboring grid cells? (TBD)

### Fallback Option

- **One-way coupling:** Fluid pushes cells only (if two-way too expensive)
- Performance testing required to determine viability

---

## Chemical Diffusion

### ⚠️ NEEDS DETAILED SPECIFICATION

**Storage and Transport:**

- Chemicals stored per-grid-cell (64³ fluid grid)
- 4 chemical types per cell
- Diffusion follows concentration gradients (PDE-based)
- Fluid currents advect/distort chemical gradients

**Diffusion Parameters (TBD):**

- Diffusion rate constant (fast equilibrium vs slow spread?)
- Integration method (explicit Euler, implicit, semi-implicit?)
- Update frequency (every tick at 50 TPS or less frequent?)

**Advection:**

- Semi-Lagrangian advection or other method?
- Advection strength relative to diffusion?
- How to balance diffusion vs advection for gameplay?

**Decay:**

- Exponential half-life, linear degradation, or configurable per chemical?
- Decay rate vary with concentration or constant?
- Different decay rates per chemical type?

**Saturation:**

- Configurable maximum concentration limit per grid cell
- Same limit for all 4 chemicals or individual limits?
- What happens when Secrocyte secretes into saturated cell (waste biomass, queue, fail)?

**Secrocyte Biomass Cost:**

- Cost per unit chemical secreted?
- Fixed cost per tick while active?
- Cost proportional to secretion rate?
- Total biomass budget for secretion?

**Cell Sensing:**

- Samples current grid cell + 26 adjacent cells (3×3×3 neighborhood)
- Simple average, weighted by distance, or maximum concentration?
- Sampling frequency: every tick or less?

**Multi-Cell Grid Cells:**

- Multiple Secrocytes in same grid cell have additive output?
- Chemical concentration calculation with multiple contributors?
- Priority system for chemical secretion?

---

## Food System

### ⚠️ NEEDS DETAILED SPECIFICATION

**Spawning System:**

- Continuous spawning at fixed rate, or initial burst then static?
- Spawn locations:
    - Random throughout world?
    - Concentrated in regions (light zones for photosynthesis analogy)?
    - At boundaries (nutrition entering system)?
    - User-configurable patterns?
- Maximum food pellet count limit (performance cap)?
- Different food types with different properties, or single generic nutrient?
- User-configurable spawn rate and patterns?

**Food Pellet Physics:**

- **If moving with currents:**
    - Actual physics objects with mass/velocity/collision detection?
    - Or just positions advected by fluid grid (cheaper)?
- Pellet collisions:
    - Collide with cells (bounce off, pushable)?
    - Pass through cells?
    - Collide with each other or overlap freely?
- Pellet size relative to cells (smaller, same, larger)?
- Affected by world boundary (soft collision like cells)?

**Consumption Mechanics:**

- Phagocyte must physically touch pellet (collision-based)?
- Or proximity-based absorption radius?
- Entire pellet consumed instantly or gradual digestion over ticks?
- Pellet mass → cell biomass conversion ratio (1:1 or efficiency factor <1)?
- Can multiple Phagocytes compete for same pellet?
- Does consumption remove pellet from spatial grid immediately?

**Pellet Lifecycle:**

- Only removed when eaten?
- Decay/despawn over time (prevents infinite accumulation)?
- Maximum lifetime for pellets?
- Visual feedback for pellet decay/freshness (color change)?

**Performance Considerations:**

- With fluid-advected pellets + 100K cells, maximum sustainable pellet count?
- LOD for pellets (distant pellets simplified or culled)?
- Spatial grid optimization for pellet-cell collision detection?
- GPU vs CPU pellet simulation?

---

## Light System

### ⚠️ NEEDS DETAILED SPECIFICATION

**Light Sources:**

- Point lights, directional lights (sun), or both?
- Light source positions:
    - Fixed in world space?
    - User can add/move/remove lights?
    - Dynamic light sources (bioluminescent cells)?
- Light color/spectrum affects Photocyte differently?
- Or just intensity scalar?
- Multiple lights or single global light?
- Light attenuation model (inverse square, linear, constant)?

**Light Propagation:**

- Raytracing from sources through grid?
- Diffusion simulation (light as diffusing quantity)?
- Raycasting from sources to each grid cell?
- Calculated once at setup or dynamically updated?
- Update frequency (every tick, less often since cells move slowly)?

**Occlusion by Cells:**

- Shadow casting mechanics:
    - Hard shadows (binary occluded/lit)?
    - Soft shadows (partial occlusion)?
- Cell opacity:
    - All cells equally opaque?
    - Cell type/size affects opacity?
    - Transparency based on cell properties?
- Clustered organisms:
    - Create dense shadow regions (realistic)?
    - Individual cell shadows?
    - Cumulative occlusion model?
- Does occlusion affect fluid simulation (cold spots in shadows → convection)?

**Grid Storage:**

- Light intensity per grid cell (scalar value 0-1)?
- Directional information (vector indicating light source direction)?
- Color channels (RGB) or single intensity?
- Multiple light accumulation method?

**Photocyte Energy Gain:**

- Linear relationship (2× light = 2× energy)?
- Diminishing returns (logarithmic curve)?
- Surface area exposure matters?
- Does cell orientation matter (facing light vs facing away)?
- Or just presence in lit grid cell sufficient?

**Performance:**

- Light calculation frequency vs cell movement
- GPU-accelerated light propagation?
- Pre-computed light maps vs dynamic calculation?
- LOD for light calculation (distant regions less accurate)?

---

## Dissolved Nutrients

### Grid-Based Storage

- Stored in 64³ fluid grid alongside chemicals
- Concentration per grid cell
- Advected by fluid currents
- Diffuse following concentration gradients

### Chemocyte Consumption

- ⚠️ Rate based on surface area, concentration gradient, or fixed? (TBD)
- Depletes local grid cell concentration
- Can starve if nutrients too low

### Nutrient Sources

- Cell death releases nutrients
- Nitrocyte cells generate nutrients
- External spawning sources? (TBD)
- Food pellet dissolution? (TBD)

### Saturation

- Adjustable saturation limit per grid cell
- Prevents infinite accumulation
- Overflow behavior? (TBD)

---

## Gas System

### Dissolved Gases

- O2, CO2, and other gases dissolved in fluid
- Required for certain cell types to function
- Diffuse and advect like chemicals
- Concentration affects cell viability (TBD)

### Gas Exchange

- Cells consume/produce gases? (TBD)
- Photosynthesis analog (Photocyte produces O2)? (TBD)
- Respiration (cells consume O2)? (TBD)

### Surface Exchange

- Gases enter/exit at world boundary? (TBD)
- Equilibrium with external atmosphere? (TBD)
- Closed system vs open system? (TBD)

---

## Camera System

### ✅ 6DOF Camera (Space Engineers-like)

- **WASD:** Move through world
- **Mouse Look:** Rotate view
- **Q/E:** Roll left/right
- **Implementation:** Complete

### ✅ Orbital Camera (Blender/Unity-like)

- **Mouse Drag:** Orbit around focus point
- **Pan:** Shift + Mouse Drag
- **Zoom:** Mouse wheel
- **Implementation:** Complete

### ✅ Compass

- Shows camera orientation
- 3D compass for global orientation
- **Implementation:** Complete

---

## Genome Editor

### ✅ Visual Editor

**Implemented features:**

- Mode list with hierarchical naming (M 0, M 0.1, M 1, etc.)
- Cell type dropdown selection
- 3D sphere controls for child positioning:
    - Parent split angle (pitch/yaw dials)
    - Child settings (pitch/yaw spheres)
    - 11.25° snapping option
- Parameter sliders:
    - Split mass
    - Split interval
    - Nutrient priority
    - Max/min connections
    - "Prioritize When Low" toggle
- Keep adhesion checkboxes (Child A/B)
- Child mode references (dropdowns)
- Make adhesion toggle

### ✅ Genome Graph

**Implemented features:**

- Node-based visualization of mode hierarchy
- Color-coded nodes by mode
- Connection lines showing child references
- Interactive controls:
    - **Shift+Click:** Add mode
    - **Shift+Right-Click:** Remove mode
    - **Right-Click link:** Self-reference toggle
    - **Middle Drag:** Pan view
- Grid background for spatial reference
- Node displays:
    - Mode name
    - Parent status
    - Cell type
    - Split interval
    - Child A/B references

### Save/Load

- "Save Genome" button
- "Load Genome" button
- Human-readable text format (delta from defaults)

---

## Tool System

### ⚠️ NEEDS COMPREHENSIVE SPECIFICATION

**Radial Menu System:**

- **Alt key:** Opens radial menu for tool/option selection
- **1-0 hotkeys:** Context-sensitive quick actions based on active tool
- Visual feedback:
    - Smooth animation for menu open/close
    - Highlight on hover
    - Display hotkey numbers on menu items
- Nested radial menus or single-level only? (TBD)

**Tool Modes:**

#### Select Tool

- Click to select single cell
- Drag box for multi-select? (TBD)
- Shift-click for adding to selection? (TBD)
- **Hotkeys 1-5:** Selection modes? (TBD)

#### Drag Tool

- Click-and-drag physics interaction
- Apply forces or teleport cells? (TBD)
- What happens to adhesions during drag? (TBD)
- Temporary constraints or impulses? (TBD)

#### Add Cell Tool

- Click to place cell
- **Hotkeys 1-0:** Quick-load saved genomes from hotbar
- Position/orientation determination? (TBD)
- Start with default mass or split-ready mass? (TBD)

#### Edit Tool

- Select cell → modify properties in inspector panel? (TBD)
- Live editing during simulation or pause-only? (TBD)

#### Remove Tool

- Click to delete single cell
- Adhesion handling: break cleanly or transfer to neighbors? (TBD)

#### Sample Tool

- Click cell to load genome into editor
- Copy entire organism's genome or just cell's mode? (TBD)

**Cell Selection and Raycasting:**

- Ray generation from mouse cursor through camera
- Intersection test optimization using spatial grid (avoid testing all 200K cells)
- Selection highlighting:
    - Outline shader? (TBD)
    - Color tint? (TBD)
    - Wireframe overlay? (TBD)
- Multi-selection behavior for organism groups
- Selection persistence across simulation ticks

**Cell Editing Capabilities:**

- Editable properties: mass, velocity, position, rotation, mode? (TBD)
- Change cell genome/mode mid-simulation (instant stem cell conversion)? (TBD)
- Undo/redo system for manual edits? (TBD)
- Validation to prevent invalid states (negative mass, broken adhesions)

**Interaction During Simulation:**

- Tools usable while simulation running or pause-only? (TBD)
- If running: manual forces interact with physics how? (additive or override)? (TBD)
- Dragging creates temporary constraints or applies impulses? (TBD)

---

## Genome Hotbar

### ⚠️ NEEDS SPECIFICATION

**Visual Hotbar:**

- 10 genome slots (1-0 hotkeys)
- Mini thumbnails of saved genomes
- Empty slots show nothing (no placeholder graphics)
- Active selection highlighted on hotbar
- Persistent display at bottom/top of screen? (TBD)

**Screenshot Capture Tool:**

- Camera positioning for capture:
    - Auto-frame organism (fit to view)? (TBD)
    - Manual positioning? (TBD)
    - Standard view angle? (TBD)
- Screenshot resolution and format (PNG recommended)
- Storage:
    - Alongside genome file? (TBD)
    - Embedded in genome metadata? (TBD)
- Lighting/background for screenshot:
    - Clean white background? (TBD)
    - Current scene lighting? (TBD)
    - Configurable? (TBD)

**Genome Save UI:**

- Name input field (required)
- Description/notes field (optional metadata)
- Screenshot preview before saving
- Save to hotbar slot directly or separate library? (TBD)
- Can genomes be saved to disk separate from hotbar slots? (YES)
- Import/export genome files from community/other users? (TBD)

**Hotkey Preview:**

- Hovering over hotbar slot:
    - Show larger preview image + genome name
    - Display basic stats (cell count after N divisions, cell types used)? (TBD)
- Preview updates when hotkey pressed but before placing cell

**Library Management:**

- How many total saved genomes beyond 10 hotbar slots? (TBD)
- Organize into folders/categories? (TBD)
- Search/filter functionality? (TBD)
- Genome versioning (track iterations)? (TBD)

---

## Rendering Pipeline

### ⚠️ NEEDS COMPREHENSIVE SPECIFICATION

**LOD Strategy:** Currently: Distance-based with 4 mesh subdivision levels

**Questions requiring review:**

- Should LOD use distance only, screen-space size, or both?
- Define the 4 LOD levels explicitly:
    - LOD 0: Full icosphere (subdivision level?)
    - LOD 1: Medium poly count (subdivision level?)
    - LOD 2: Low poly count (subdivision level?)
    - LOD 3: Billboard impostor / point sprite
    - LOD 4: Culled entirely
- Transition distances for each LOD level:
    - LOD 0→1 at distance X?
    - LOD 1→2 at distance Y?
    - LOD 2→3 at distance Z?
    - Cull beyond distance W?
- Hysteresis to prevent LOD popping:
    - Transition zones where both LODs blend?
    - Delay before switching to prevent flickering?
- Adhesion rendering LOD:
    - Thick cylinders → thin lines → invisible at distance?
    - Same distance thresholds as cells?

**Culling Pipeline Order:** Need to define execution sequence:

1. Frustum culling (eliminate off-screen cells)
2. Occlusion culling (eliminate hidden cells)
3. Cluster/coarse culling (batch processing)
4. Distance culling (eliminate far cells)

**Occlusion Culling Implementation:**

- GPU-based depth pyramid / Hierarchical Z-Buffer?
- CPU-based bounding volume tests?
- GPU compute shader culling (most efficient for 200K cells)?
- Two-pass rendering:
    - Pass 1: Depth pre-pass (front-to-back)
    - Pass 2: Shaded rendering with occlusion test
- Performance cost vs benefit at 100K+ cells?

**Front-to-Back Rendering vs Transparency:**

- Depth pre-pass incompatible with transparency
- Are cells transparent or opaque?
- If transparent:
    - Order-independent transparency (OIT)?
    - Sorted transparency (performance cost)?
    - Disable depth pre-pass for transparent cells?

**Impostor Implementation Choice:** Need to benchmark and choose:

_Option A: Spherical Billboards_

- Quad always facing camera
- Texture with shaded sphere
- Simple, fast, less accurate
- Normal map for lighting?

_Option B: Raymarching in Fragment Shader_

- Per-pixel sphere raycast
- Most accurate rendering
- Compute cost per pixel
- Best visual quality
- Spatial grid for acceleration?

_Option C: Point Sprites_

- GPU-native point rendering
- Fastest but least control
- Limited size/shading options

**Performance Comparison Required:**

- Which delivers better FPS at 100K+ cells?
- Quality vs performance tradeoff
- Hybrid approach (billboards near, points far)?

**Impostor Lighting:**

- Do impostors cast shadows?
- Receive shadows from other objects?
- Normal generation for lighting calculations?
- PBR materials or simple shading?

**Smooth Cell Deformation:** Currently marked as "peak but questionable performance"

_Potential Approaches:_

- Metaball-style marching cubes mesh generation
- GPU-accelerated dual contouring
- Real-time mesh generation per frame
- Blend between nearby cells for organic look

_Questions:_

- Real-time mesh generation cost vs visual improvement
- GPU compute shader for mesh generation?
- Update frequency (every frame too expensive?)
- Should this be abandoned entirely for performance?
- Optional "ultra quality" rendering mode only?
- Pre-computed LOD meshes for common configurations?

**GPU-Driven Rendering:**

- Entire pipeline on GPU (no CPU→GPU readback)?
- Indirect draw commands?
- GPU frustum culling in compute shader?
- GPU LOD selection in compute shader?

---

## Debug Information

### Performance Metrics

- [ ] FPS (frames per second)
- [ ] TPS (ticks per second)
- [ ] Current simulation mode
- [ ] Execution time breakdown:
    - Physics step time
    - Collision detection time
    - Adhesion calculation time
    - Fluid simulation time
    - Rendering time
    - Memory usage

### World Information

- [ ] 3D compass for global orientation (✅ implemented)
- [ ] Grid and axes for spatial reference
- [ ] World boundary visualization (✅ implemented)
- [ ] Cell count (current / maximum)
- [ ] Organism count
- [ ] Adhesion count

### Cell Information

When cell selected:

- [ ] All properties of selected cell:
    - Mode name and index
    - Cell type
    - Position (x, y, z)
    - Velocity (x, y, z)
    - Rotation (quaternion or euler)
    - Angular velocity
    - Mass / biomass
    - Age
    - Adhesion count
    - Parent cell reference
    - Genome reference
    - Priority value
- [ ] Orientation gizmos (XYZ axes)
- [ ] Adhesion anchor gizmos
- [ ] Split plane gizmos

---

## Simulation Control

### Cell Count Limits

- Per-mode configurable maximums
- Preview: 256
- CPU: 5K-10K (user configurable)
- GPU: 100K-200K (user configurable)

### Time Control

- [ ] Pause/Resume
- [ ] Physics tickrate decoupled from rendering framerate
- [ ] Set target tickrate (default 50 TPS)
- [ ] Set target framerate (default 60 FPS)
- [ ] Priority: ticks over frames down to 1 FPS minimum
- [ ] Configurable timestep size (default 0.02s)
- [ ] Variable speed controls (GPU mode only)
- [ ] Manual scrubber control (Preview and CPU modes)

### Mode Transitions

- Transitioning between modes restarts simulation
- Save/load states preserve progress
- Manual cell placement required for genome injection

---

## Save/Load System

### Simulation Save Files

**Format:** Binary for performance and size

**Saved Data:**

- Cell data (~80 bytes × cell count):
    - Positions, velocities, rotations, angular velocities
    - Masses, ages, genome indices
    - Mode references, parent references
- Adhesion data (~40 bytes × adhesion count):
    - Anchor data, rest lengths
    - Cell A/B references, original sides
- Grid state (~40 bytes × 262K grid cells):
    - Fluid velocities
    - Dissolved nutrients
    - Chemical concentrations
    - Light intensity
    - Temperature (if implemented)
    - Gas concentrations
- Genome library:
    - Deduplicated genomes with mode data
    - ~24KB × unique genome count

**Estimated File Sizes:**

- 100K cell GPU sim: ~40-50MB uncompressed
- Potential compression: 10-20MB with zip compression

**Save Features:**

- Binary format (fast, compact)
- Delta compression for sequential saves? (TBD)
- Separate autosave slots vs manual saves? (TBD)
- Save metadata:
    - Timestamp
    - Tick count
    - Cell count
    - Thumbnail image? (TBD)
    - User notes? (TBD)

**Load Features:**

- Validation that save is compatible with current version
- Warning on version mismatch
- Migration tool for breaking changes
- Handle corrupted saves gracefully

### Genome Export Files

**Format:** Human-readable text (TOML, JSON, or custom)

**Delta Encoding:**

- Only display modes changed from defaults
- Compact, readable, easy to share
- Version control friendly

**Upgrade Utility:**

- Comes with warning about potential organism breakage
- Migrates genome format across versions
- Shows diff of changes
- Backup original before upgrade

**Community Sharing:**

- Import/export between users
- Genome file format specification
- Validation tool for imported genomes
- Sandbox test before using in main sim

---

## Rendering - Cell Visualization

### Debug Rendering

- [ ] Colored icospheres for cells
- [ ] Lines for adhesions
- [ ] Wireframe mode toggle
- [ ] Overlay mode (wireframe + solid)

### Fancy Cell Rendering

- [ ] ⚠️ Smooth cell deformation (performance questionable)
    - Metaball-style marching cubes?
    - GPU mesh generation?
    - Update frequency?
    - Optional ultra quality mode?

### Cell Mesh LOD

- LOD 0: High-poly icosphere
- LOD 1: Medium-poly icosphere
- LOD 2: Low-poly icosphere
- LOD 3: Billboard impostor
- Distance-based transitions

### Materials

- PBR materials? (TBD)
- Simple phong shading? (TBD)
- Cell color from genome
- Transparency/opacity? (TBD)
- Emissive cells (bioluminescence)? (TBD)

---

## Rendering - Adhesions

### LOD Strategy (TBD)

- Close: Thick cylinders with proper geometry
- Medium: Thin lines (GL_LINES)
- Far: Invisible / culled
- Distance thresholds aligned with cell LOD

### Visual Style

- Color coding:
    - By strength? (TBD)
    - By type? (TBD)
    - By strain (stressed adhesions)? (TBD)
- Thickness based on adhesion strength? (TBD)
- Animated when breaking? (TBD)

---

## Performance Budget Estimates

### GPU Simulation (200K cells)

**Target: <0.2Fr = <40ms per tick**

Estimated breakdown:

- Spatial grid update: ~5ms
- Collision detection: ~10ms
- Adhesion forces: ~10ms
- Integration (Verlet): ~5ms
- Cell division allocation: ~5ms
- Fluid simulation: ~15ms
- Chemical diffusion: ~10ms
- Light calculation: ~5ms
- **Total: ~65ms** ❌ Over budget

⚠️ **Optimization required** - likely need:

- Adaptive fluid updates
- Less frequent chemical diffusion
- Simplified light calculation
- GPU-driven LOD earlier

### Rendering (200K cells at 60 FPS)

**Target: <16.67ms per frame**

Estimated breakdown:

- Culling (frustum + occlusion): ~3ms
- LOD selection: ~2ms
- Instanced rendering: ~8ms
- UI rendering: ~2ms
- Post-processing: ~2ms
- **Total: ~17ms** ⚠️ Tight budget

May need to target 30 FPS for complex scenes

---

## Future Considerations

### Advanced Features (Not in Current Scope)

- Temperature simulation and convection
- Electrical signaling (action potentials)
- Sound propagation (Audiocyte implementation)
- Advanced neural networks (complex Neurocyte behaviors)
- Genetic algorithms and evolution
- Multi-species ecosystems
- Predator-prey dynamics
- Symbiosis mechanics
- Environmental zones (hot/cold, light/dark)
- Seasonal cycles
- Day/night light cycles

### Platform Targets

- Desktop (Windows, Linux, macOS)
- Web (WASM + WebGPU)
- Mobile (iOS, Android) - requires significant optimization

### Multiplayer Considerations

- Shared simulation spaces
- Organism trading/sharing
- Collaborative editing
- Competitive challenges
- Deterministic networking

---

## Open Questions Summary

### Critical Path (Must Resolve Before Implementation)

1. **Large penetration handling** - collision stability strategy
2. **Behavior/control system** - complete architecture needed
3. **Sensor detection** - range, occlusion, performance optimization
4. **Chemical diffusion** - rates, costs, update frequency
5. **Food system** - spawning, physics, consumption mechanics
6. **Light system** - propagation, occlusion, Photocyte energy
7. **Rendering pipeline** - LOD, culling, impostor choice

### High Priority (Needed Soon)

8. **Tool system** - complete interaction design
9. **Genome hotbar** - UI and workflow
10. **Mutation system** - if evolution is core feature

### Medium Priority (Can Defer)

11. **Performance budget** - actual profiling needed
12. **Save format details** - metadata, compression
13. **Community features** - sharing, validation

### Low Priority (Polish)

14. **Advanced rendering** - deformation, materials
15. **Audio system** - Audiocyte implementation
16. **Platform ports** - web, mobile

---

## Document Change Log

**Version 1.0** - Initial comprehensive design document

- Compiled all original requirements
- Added detailed considerations from design review session
- Marked areas needing further review
- Created glossary with internal links
- Organized into logical sections
- Estimated performance budgets
- Identified critical path items