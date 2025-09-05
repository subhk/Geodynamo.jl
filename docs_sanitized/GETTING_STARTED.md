#GettingStartedwithGeodynamo.jl

WelcometoGeodynamo.jl!Thisguidewillhelpyougetupandrunningwithgeodynamosimulationsusingsphericalharmonictransformsandflexibleboundaryconditions.

##QuickStart

###Installation

```julia
usingPkg
Pkg.add("Geodynamo")
```

###YourFirstSimulation

```julia
usingGeodynamo

#1.Createabasicconfiguration
config=create_optimized_config(32,32,nlat=64,nlon=128)

#2.Setupthesimulationdomain
domain=create_radial_domain(0.35,1.0,64)#Innerradius,outerradius,radialpoints

#3.Createtemperaturefield
temp_field=create_shtns_temperature_field(Float64,config,domain)

#4.Setsimpleboundaryconditions
apply_netcdf_temperature_boundaries!(temp_field,
create_hybrid_temperature_boundaries(
(:uniform,4000.0),#HotCMB
(:uniform,300.0),#Coolsurface
config
)
)

println("Yourfirstgeodynamosimulationisready!")
```

##TableofContents

1.[CoreConcepts](#core-concepts)
2.[SettingUpSimulations](#setting-up-simulations)
3.[BoundaryConditions](#boundary-conditions)
4.[RunningSimulations](#running-simulations)
5.[Examples](#examples)
6.[Troubleshooting](#troubleshooting)
7.[AdvancedTopics](#advanced-topics)

---

##CoreConcepts

###WhatisGeodynamo.jl?

Geodynamo.jlisaJuliapackageforsimulatingEarth'smagneticfieldgenerationthroughthegeodynamoprocess.Ituses:

-**SphericalHarmonicTransforms(SHTns)**:Efficientspectralmethodsforsphericalgeometry
-**FlexibleBoundaryConditions**:SupportforNetCDFdatafilesandprogrammaticpatterns
-**HighPerformance**:CPU-optimizedwithSIMDvectorizationandthreading
-**MPIParallelization**:Scalabletolargeclusters

###KeyComponents

-**TemperatureField**:Controlsbuoyancyandconvection
-**CompositionalField**:Trackslight/heavyelementdistribution
-**MagneticField**:Thedynamo-generatedmagneticfield
-**VelocityField**:Fluidmotiondrivingthedynamo

###CoordinateSystem

-**Sphericalcoordinates**:(r,θ,φ)whereθiscolatitude[0,π],φislongitude[0,2π]
-**Radialdomain**:Frominnercoreboundarytocore-mantleboundary
-**Spectralrepresentation**:SphericalharmonicsY_l^mforangulardependence

---

##SettingUpSimulations

###Step1:Configuration

CreateanSHTnsconfigurationspecifyingresolutionandoptimization:

```julia
usingGeodynamo

#Basicconfiguration
config=create_optimized_config(
32,32,#lmax,mmax(sphericalharmonicresolution)
nlat=64,nlon=128,#Physicalgridresolution
use_threading=true,#EnableCPUthreading
use_simd=true#EnableSIMDvectorization
)
```

####ResolutionGuidelines

|ProblemSize|lmax/mmax|nlat×nlon|UseCase|
|-------------|-----------|-------------|----------|
|Learning/Testing|16-32|32×64|Quickexperiments|
|Research|64-128|128×256|Productionsimulations|
|High-Resolution|256+|512×1024|Publicationquality|

###Step2:DomainSetup

Definetheradialdomain(sphericalshell):

```julia
#Earth-likeparameters
inner_radius=0.35#Innercoreboundary(normalized)
outer_radius=1.0#Core-mantleboundary(normalized)
nr=64#Numberofradialgridpoints

domain=create_radial_domain(inner_radius,outer_radius,nr)
```

###Step3:CreateFields

Initializethephysicalfieldsforyoursimulation:

```julia
#Temperaturefield(drivesconvection)
temp_field=create_shtns_temperature_field(Float64,config,domain)

#Compositionalfield(optional,forchemicalconvection)
comp_field=create_shtns_composition_field(Float64,config,domain)

#Velocityfield(fluidmotion)
vel_field=create_shtns_velocity_fields(Float64,config,domain)

#Magneticfield(dynamooutput)
mag_field=create_shtns_magnetic_fields(Float64,config,domain)
```

---

##BoundaryConditions

Geodynamo.jloffersflexibleboundaryconditionspecification.Youcanmixandmatchdifferentapproaches:

###Option1:SimpleProgrammaticBoundaries

Perfectforlearningandtesting:

```julia
#Createuniformtemperatureboundaries
temp_boundaries=create_hybrid_temperature_boundaries(
(:uniform,4000.0),#Innerboundary:4000K(hotCMB)
(:uniform,300.0),#Outerboundary:300K(coolsurface)
config
)

#Applytotemperaturefield
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)
```

###Option2:SphericalHarmonicPatterns

Addrealisticspatialstructure:

```julia
#CMBwithY₁₁perturbation,uniformsurface
temp_boundaries=create_hybrid_temperature_boundaries(
(:y11,4000.0,Dict("amplitude"=>200.0)),#CMB:base+Y₁₁pattern
(:uniform,300.0),#Surface:uniform
config
)
```

###Option3:RealisticPatterns

Usephysicalpatternslikeplumes:

```julia
#HotplumeatCMB,uniformsurface
temp_boundaries=create_hybrid_temperature_boundaries(
(:plume,4200.0,Dict(
"center_theta"=>π/3,#Plumelocation(colatitude)
"center_phi"=>π/4,#Plumelocation(longitude)
"width"=>π/8#Plumewidth
)),
(:uniform,300.0),
config
)
```

###Option4:NetCDFDataFiles

Useobservationalornumericalmodeldata:

```julia
#Firstcreatesamplefiles(one-timesetup)
include("examples/create_sample_netcdf_boundaries.jl")

#LoadfromNetCDFfiles
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc")
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)
```

###Option5:Hybrid(MixNetCDF+Programmatic)

Combinedatafileswithanalyticalpatterns:

```julia
#NetCDFCMBdata+simpleuniformsurface
temp_boundaries=create_hybrid_temperature_boundaries(
"realistic_cmb.nc",#ComplexCMBfromdatafile
(:uniform,300.0),#Simpleuniformsurface
config
)
```

###AvailableProgrammaticPatterns

|Pattern|Description|Parameters|
|---------|-------------|------------|
|`:uniform`|Constantvalue|amplitude|
|`:y11`|Y₁₁sphericalharmonic|amplitude|
|`:y20`|Y₂₀zonalharmonic|amplitude|
|`:plume`|Gaussianplume|center_theta,center_phi,width|
|`:hemisphere`|Half-spherepattern|axis("x","y","z")|
|`:dipole`|Dipolar(Y₁₀)|amplitude|
|`:checkerboard`|Alternatingblocks|nblocks_theta,nblocks_phi|
|`:custom`|Yourfunction|function|

---

##RunningSimulations

###BasicTimeStepping

```julia
#Simulationparameters
dt=0.001#Timestep
nsteps=1000#Numberofsteps
output_freq=100#OutputeveryNsteps

#Timesteppingloop
forstepin1:nsteps
current_time=step*dt

#Updatetime-dependentboundaries(ifapplicable)
update_temperature_boundaries_from_netcdf!(temp_field,temp_boundaries,step,dt)

#Computenonlinearterms
compute_temperature_nonlinear!(temp_field,vel_field)

#Timestep(simplified-actualimplementationmorecomplex)
#solve_implicit_step!(temp_field,dt)

#Outputresults
ifstep%output_freq==0
println("Step$step,time=$(current_time)")
#write_fields!(...)#Savetofiles
end
end
```

###MonitoringPerformance

Tracksimulationperformance:

```julia
#Resetperformancecounters
reset_performance_stats!()

#Runsimulationwithmonitoring
@timed_transformbegin
forstepin1:nsteps
#...simulationsteps
end
end

#Viewperformancereport
print_performance_report()
```

---

##Examples

###Example1:BasicThermalConvection

```julia
usingGeodynamo

functionbasic_thermal_convection()
println("BasicThermalConvectionExample")

#Setup
config=create_optimized_config(32,32,nlat=64,nlon=128)
domain=create_radial_domain(0.35,1.0,64)

#Createfields
temp_field=create_shtns_temperature_field(Float64,config,domain)

#Simpleboundaries:hotbottom,coldtop
temp_boundaries=create_hybrid_temperature_boundaries(
(:uniform,4000.0),#HotCMB
(:uniform,300.0),#Coolsurface
config
)

#Applyboundaries
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)

#Addsmallperturbationtoinitiateconvection
set_temperature_ic!(temp_field,:random_perturbation,domain)

println("Basicthermalconvectionsetupcomplete")
print_boundary_info(temp_boundaries)
end

basic_thermal_convection()
```

###Example2:Plume-DrivenConvection

```julia
functionplume_convection()
println("Plume-DrivenConvectionExample")

config=create_optimized_config(64,64,nlat=128,nlon=256)
domain=create_radial_domain(0.35,1.0,64)

temp_field=create_shtns_temperature_field(Float64,config,domain)

#HotplumeatCMB
temp_boundaries=create_hybrid_temperature_boundaries(
(:plume,4200.0,Dict(
"center_theta"=>π/3,#60°fromnorthpole
"center_phi"=>0.0,#Primemeridian
"width"=>π/6#30°width
)),
(:uniform,300.0),#Uniformcoolsurface
config
)

apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)

println("Plumeconvectionsetupcomplete")

#Showboundarystatistics
inner_stats=get_boundary_statistics(temp_boundaries.inner_boundary)
println("CMBtemperaturerange:[$(round(inner_stats["min"])),$(round(inner_stats["max"]))]K")
end

plume_convection()
```

###Example3:UsingRealData

```julia
functionrealistic_boundaries()
println("RealisticBoundaryConditionsExample")

#CreatesampleNetCDFfilesifneeded
if!isfile("cmb_temp.nc")
include("examples/create_sample_netcdf_boundaries.jl")
end

config=create_optimized_config(64,64,nlat=128,nlon=256)
domain=create_radial_domain(0.35,1.0,64)

#LoadtemperatureboundariesfromNetCDFfiles
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc")

#Loadcompositionboundaries
comp_boundaries=load_composition_boundaries("cmb_composition.nc","surface_composition.nc")

#Createfields
temp_field=create_shtns_temperature_field(Float64,config,domain)
comp_field=create_shtns_composition_field(Float64,config,domain)

#Applyboundaries
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)
apply_netcdf_composition_boundaries!(comp_field,comp_boundaries)

println("Realisticboundariesloaded")
print_boundary_info(temp_boundaries)
print_boundary_info(comp_boundaries)
end

realistic_boundaries()
```

###Example4:Time-DependentBoundaries

```julia
functiontime_dependent_example()
println("Time-DependentBoundariesExample")

config=create_optimized_config(32,32,nlat=64,nlon=128)
domain=create_radial_domain(0.35,1.0,64)

#Createrotatingplumepattern
rotating_inner=create_time_dependent_programmatic_boundary(
:plume,config,(0.0,10.0),50,#50timestepsover10timeunits
amplitude=4200.0,
parameters=Dict(
"width"=>π/6,
"center_theta"=>π/3,
"time_factor"=>2π,#Onefullrotation
)
)

#Createboundaryset
temp_boundaries=BoundaryConditionSet(
rotating_inner,
create_programmatic_boundary(:uniform,config,amplitude=300.0),
"temperature",
time()
)

temp_field=create_shtns_temperature_field(Float64,config,domain)

#Simulatetimeevolution
dt=0.2
forstepin1:10
current_time=step*dt
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries,current_time)
println("Step$step:Appliedboundariesattime$current_time")
end

println("Time-dependentboundaryexamplecomplete")
end

time_dependent_example()
```

---

##Troubleshooting

###CommonIssuesandSolutions

####Issue:"NetCDFfilenotfound"
```
ERROR:NetCDFfilenotfound:cmb_temp.nc
```
**Solution**:Createsamplefilesfirst:
```julia
include("examples/create_sample_netcdf_boundaries.jl")
```

####Issue:"Gridsizemismatch"
```
ERROR:Innerboundarynlat(32)!=confignlat(64)
```
**Solution**:Either:
1.Matchgridsizes:`create_optimized_config(32,32,nlat=32,nlon=64)`
2.Orrelyonautomaticinterpolation(maybeslower)

####Issue:"SHTnsKitnotfound"
```
ERROR:PackageSHTnsKitnotfound
```
**Solution**:Installlocaldependency:
```julia
usingPkg
Pkg.develop(path="../SHTnsKit.jl")#Adjustpathasneeded
```

####Issue:Performancewarnings
```
WARNING:Typeinstabilitydetected
```
**Solution**:Checkfieldtypesanduseconsistentprecision:
```julia
temp_field=create_shtns_temperature_field(Float64,config,domain)#ExplicitFloat64
```

###DebuggingTips

1.**StartSimple**:Beginwithlowresolutionanduniformboundaries
2.**CheckBoundaries**:Use`print_boundary_info()`toinspectloadeddata
3.**MonitorPerformance**:Use`@timed_transform`and`print_performance_report()`
4.**ValidateData**:Use`get_boundary_statistics()`tocheckloadedvalues

###GettingHelp

-**Documentation**:Checkthedocs/directoryfordetailedguides
-**Examples**:Runexamples/scriptstoseeworkingcode
-**Issues**:ReportbugsontheGitHubrepository
-**Community**:Askquestionsindiscussions

---

##AdvancedTopics

###MPIParallelization

Forlargesimulations,useMPI:

```bash
mpirun-np4julia--project=.my_simulation.jl
```

```julia
usingMPI
MPI.Init()

comm=MPI.COMM_WORLD
rank=MPI.Comm_rank(comm)
nprocs=MPI.Comm_size(comm)

ifrank==0
println("Runningon$nprocsprocesses")
end

#...restofsimulation
```

###CustomBoundaryPatterns

Createyourownboundarypatterns:

```julia
#Definecustomfunction
functionmy_boundary_pattern(theta,phi)
#Yourmathematicalexpressionhere
returnsin(3*theta)*cos(2*phi)+0.5*cos(theta)^2
end

#Useinboundaryspecification
boundaries=create_hybrid_temperature_boundaries(
(:custom,4000.0,Dict("function"=>my_boundary_pattern)),
(:uniform,300.0),
config
)
```

###High-PerformanceTips

1.**Useappropriateprecision**:Float64foraccuracy,Float32forspeed
2.**Enableoptimizations**:`use_threading=true`,`use_simd=true`
3.**Chooseresolutionwisely**:Balanceaccuracyvs.computationalcost
4.**Profileyourcode**:Use`@timed_transform`toidentifybottlenecks

###CreatingNetCDFFiles

Foryourowndata,createNetCDFfileswiththisstructure:

```julia
usingNCDatasets

NCDataset("my_boundary.nc","c")dods
#Definedimensions
defDim(ds,"lat",nlat)
defDim(ds,"lon",nlon)

#Definecoordinates
defVar(ds,"theta",Float64,("lat",))#Colatitude[0,π]
defVar(ds,"phi",Float64,("lon",))#Longitude[0,2π]

#Definedatavariable
defVar(ds,"temperature",Float64,("lat","lon"),
attrib=Dict("units"=>"K","long_name"=>"Temperature"))

#Writedata
ds["theta"][:]=your_theta_data
ds["phi"][:]=your_phi_data
ds["temperature"][:]=your_temperature_data
end
```

---

##What'sNext?

1.**RuntheExamples**:Startwiththeprovidedexamplescripts
2.**ModifyParameters**:Experimentwithdifferentresolutionsandboundaryconditions
3.**AddPhysics**:Includemagneticfieldgenerationandcompositionalconvection
4.**ScaleUp**:MovetohigherresolutionandMPIparallelization
5.**AnalyzeResults**:Usebuilt-inanalysistoolsforscientificinsights

###LearningPath

1.**Beginner**:Startwithuniformboundariesandlowresolution
2.**Intermediate**:UseNetCDFfilesandsphericalharmonicpatterns
3.**Advanced**:Createcustompatternsandtime-dependentboundaries
4.**Expert**:Developnewphysicsmodulesandoptimizationtechniques

---

##QuickReferenceCard

###EssentialCommands

```julia
#Configuration
config=create_optimized_config(lmax,mmax,nlat=nlat,nlon=nlon)
domain=create_radial_domain(ri,ro,nr)

#Fields
temp_field=create_shtns_temperature_field(Float64,config,domain)

#Boundaries-Programmatic
boundaries=create_hybrid_temperature_boundaries((:pattern,amplitude),(:pattern,amplitude),config)

#Boundaries-NetCDF
boundaries=load_temperature_boundaries("inner.nc","outer.nc")

#Applyboundaries
apply_netcdf_temperature_boundaries!(temp_field,boundaries)

#Performancemonitoring
reset_performance_stats!()
@timed_transformbegin
#simulationcode
end
print_performance_report()
```

###PatternQuickReference

|Pattern|Example|Description|
|---------|---------|-------------|
|`:uniform`|`(:uniform,300.0)`|Constantvalue|
|`:y11`|`(:y11,4000.0,Dict("amplitude"=>200))`|Y₁₁+perturbation|
|`:plume`|`(:plume,4200.0,Dict("width"=>π/6))`|Gaussianplume|
|`:dipole`|`(:dipole,1000.0)`|Dipolarpattern|

Welcometogeodynamomodeling!