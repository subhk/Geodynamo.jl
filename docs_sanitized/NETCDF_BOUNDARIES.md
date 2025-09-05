#NetCDFBoundaryConditionsinGeodynamo.jl

ThisdocumentdescribeshowtouseNetCDFfilestospecifytemperatureandcompositionalboundaryconditionsinGeodynamo.jlsimulations.

##Overview

Geodynamo.jlsupportsreadingboundaryconditionsfromNetCDFfiles,providing:

-**Flexibleboundaryspecification**:Supportforbothinner(CMB)andouter(surface)boundaries
-**Time-dependentboundaries**:Evolvingboundaryconditionsduringsimulation
-**Automaticinterpolation**:HandlesgridmismatchesbetweenNetCDFfilesandsimulationgrid
-**Comprehensivevalidation**:Ensurescompatibilityanddataintegrity
-**Easyintegration**:Seamlessincorporationintoexistingsimulationworkflows

##QuickStart

```julia
usingGeodynamo

#Loadtemperatureboundaryconditionsfromseparatefiles
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc")

#Loadcompositionalboundaryconditions
comp_boundaries=load_composition_boundaries("cmb_composition.nc","surface_composition.nc")

#Applytosimulationfields
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)
apply_netcdf_composition_boundaries!(comp_field,comp_boundaries)
```

##NetCDFFileFormatRequirements

###FileStructure

EachboundaryNetCDFfileshouldcontain:

####RequiredVariables
-**Datavariable**:Themainfielddata(temperature,composition,etc.)
-**Coordinatevariables**(optionalbutrecommended):
-`theta`or`lat`/`latitude`:Colatitude[0,π]orlatitude[-90,90]indegrees
-`phi`or`lon`/`longitude`:Longitude[0,2π]or[0,360]indegrees/radians
-`time`(optional):Timecoordinatefortime-dependentboundaries

####RequiredDimensions
-**Spatialdimensions**:`lat`×`lon`(2D)or`lat`×`lon`×`time`(3D)
-**Coordinatedimensions**:Matchthespatialdimensions

###ExampleNetCDFStructure

```
dimensions:
lat=64;
lon=128;
time=10;//Optionalfortime-dependentdata

variables:
doubletheta(lat);
theta:long_name="Colatitude";
theta:units="radians";

doublephi(lon);
phi:long_name="Longitude";
phi:units="radians";

doubletime(time);//Optional
time:long_name="Time";
time:units="dimensionless_time";

doubletemperature(lat,lon);//or(lat,lon,time)
temperature:long_name="Temperatureboundarycondition";
temperature:units="K";
temperature:description="CMBtemperatureboundary";
```

##CreatingNetCDFBoundaryFiles

###UsingtheSampleGenerator

Geodynamo.jlincludesasampleNetCDFgenerator:

```bash
juliaexamples/create_sample_netcdf_boundaries.jl
```

Thiscreatesexamplefiles:
-`cmb_temp.nc`/`surface_temp.nc`:Time-independenttemperatureboundaries
-`cmb_temp_timedep.nc`/`surface_temp_timedep.nc`:Time-dependenttemperatureboundaries
-`cmb_composition.nc`/`surface_composition.nc`:Compositionalboundaries

###ManualCreation

####Time-IndependentTemperatureBoundary

```julia
usingNCDatasets

#CreateCMBtemperatureboundary
NCDataset("cmb_temp.nc","c")dods
#Definedimensions
defDim(ds,"lat",64)
defDim(ds,"lon",128)

#Definecoordinates
defVar(ds,"theta",Float64,("lat",))
defVar(ds,"phi",Float64,("lon",))

#Definetemperaturevariable
temp_var=defVar(ds,"temperature",Float64,("lat","lon"),
attrib=Dict("units"=>"K",
"long_name"=>"CMBtemperature"))

#Writecoordinatedata
theta=range(0,π,length=64)
phi=range(0,2π,length=129)[1:128]#Excludeendpoint

ds["theta"][:]=theta
ds["phi"][:]=phi

#Createtemperaturepattern
temperature=zeros(64,128)
T_base=4000.0#BaseCMBtemperature[K]

for(i,th)inenumerate(theta)
for(j,ph)inenumerate(phi)
#Addsphericalharmonicperturbation
Y11=sin(th)*cos(ph)#Y₁₁mode
temperature[i,j]=T_base+200.0*Y11
end
end

#Writetemperaturedata
temp_var[:]=temperature

#Addglobalattributes
ds.attrib["title"]="CMBTemperatureBoundary"
ds.attrib["description"]="Core-mantleboundarytemperature"
end
```

####Time-DependentBoundary

```julia
NCDataset("cmb_temp_timedep.nc","c")dods
defDim(ds,"lat",64)
defDim(ds,"lon",128)
defDim(ds,"time",100)#100timesteps

#Definevariableswithtimedimension
defVar(ds,"theta",Float64,("lat",))
defVar(ds,"phi",Float64,("lon",))
defVar(ds,"time",Float64,("time",))

temp_var=defVar(ds,"temperature",Float64,("lat","lon","time"))

#Writecoordinates
ds["theta"][:]=range(0,π,length=64)
ds["phi"][:]=range(0,2π,length=129)[1:128]
ds["time"][:]=range(0,10,length=100)#Timespan

#Writetime-dependenttemperature
temperature=zeros(64,128,100)

for(k,t)inenumerate(ds["time"][:])
for(i,th)inenumerate(ds["theta"][:])
for(j,ph)inenumerate(ds["phi"][:])
#Rotatingthermalpattern
phase=2π*t/10#Completerotationevery10timeunits
Y11=sin(th)*cos(ph+phase)
temperature[i,j,k]=4000.0+200.0*Y11
end
end
end

temp_var[:]=temperature
end
```

##LoadingBoundaryConditions

###BasicLoading

```julia
usingGeodynamo

#Loadtemperatureboundaries(innerandouterfiles)
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc")

#Loadcompositionboundaries
comp_boundaries=load_composition_boundaries("cmb_comp.nc","surface_comp.nc")

#Specifyprecision(default:Float64)
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc",
precision=Float32)
```

###InspectingLoadedBoundaries

```julia
#Printdetailedinformation
print_boundary_info(temp_boundaries)

#Getstatisticalsummary
inner_stats=get_boundary_statistics(temp_boundaries.inner_boundary)
outer_stats=get_boundary_statistics(temp_boundaries.outer_boundary)

println("Innerboundarytemperaturerange:[$(inner_stats["min"]),$(inner_stats["max"])]$(inner_stats["units"])")
println("Outerboundarytemperaturemean:$(outer_stats["mean"])$(outer_stats["units"])")
```

###Validation

```julia
#CreateSHTnsconfiguration
config=create_optimized_config(32,32,nlat=64,nlon=128)

#Validatecompatibility
validate_netcdf_temperature_compatibility(temp_boundaries,config)
validate_netcdf_composition_compatibility(comp_boundaries,config)
```

##ApplyingBoundaryConditions

###BasicApplication

```julia
#Createsimulationfields
domain=create_radial_domain(0.35,1.0,64)
temp_field=create_shtns_temperature_field(Float64,config,domain)
comp_field=create_shtns_composition_field(Float64,config,domain)

#Applyboundaryconditionsatsimulationstart
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries,0.0)
apply_netcdf_composition_boundaries!(comp_field,comp_boundaries,0.0)
```

###Time-DependentUpdates

Fortime-dependentboundaries,updateduringthesimulationloop:

```julia
dt=0.001#Timestep
nsteps=10000

fortimestepin1:nsteps
current_time=timestep*dt

#Updateboundariesbasedoncurrenttime
update_temperature_boundaries_from_netcdf!(temp_field,temp_boundaries,timestep,dt)
update_composition_boundaries_from_netcdf!(comp_field,comp_boundaries,timestep,dt)

#Performsimulationstep
#...(timesteppingcode)
end
```

##GridInterpolation

Geodynamo.jlautomaticallyhandlesgridmismatchesthroughinterpolation:

###AutomaticInterpolation

```julia
#NetCDFfilehas32×64grid,simulationuses64×128
#Interpolationhappensautomaticallyduringapplication
temp_boundaries=load_temperature_boundaries("low_res_cmb.nc","low_res_surface.nc")
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)#Auto-interpolation
```

###ManualInterpolation

```julia
#Createcustomtargetgrid
target_theta=range(0,π,length=96)
target_phi=range(0,2π,length=193)[1:192]

#Interpolateboundarydata
interpolated_data=interpolate_boundary_to_grid(temp_boundaries.inner_boundary,
target_theta,target_phi,1)#Timeindex1
```

##AdvancedUsage

###CustomCoordinateNames

TheNetCDFreadersupportsflexiblecoordinatevariablenames:

```julia
#Automaticallydetectsthesecoordinatenames:
#Latitude:"theta","colatitude","colat","lat","latitude"
#Longitude:"phi","longitude","long","lon"
#Time:"time","t","time_index"

#Customcoordinatemapping(ifneeded)
custom_coords=Dict(
"theta"=>["custom_lat","my_theta"],
"phi"=>["custom_lon","my_phi"],
"time"=>["my_time"]
)

boundary_data=read_netcdf_boundary_data("custom_file.nc","temperature",
coord_names=custom_coords)
```

###FieldNameAuto-Detection

```julia
#Auto-detectmaindatavariable(looksforcommonnames)
boundary_data=read_netcdf_boundary_data("boundary_file.nc")#Emptyfieldname=auto-detect

#Explicitlyspecifyfieldname
boundary_data=read_netcdf_boundary_data("boundary_file.nc","my_temperature_field")
```

###ErrorHandling

```julia
try
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc")
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)
catche
ifisa(e,ArgumentError)
println("Boundaryvalidationfailed:$e")
#Handlevalidationerrors
elseifisa(e,BoundsError)
println("Timeindexoutofrange:$e")
#Handletimeindexingerrors
else
println("Unexpectederror:$e")
rethrow()
end
end
```

##PerformanceConsiderations

###MemoryUsage

Forlargetime-dependentdatasets:

```julia
#Checkmemoryusagebeforeloading
fileinfo=NCDataset("large_timedep_boundary.nc","r")dods
println("Timesteps:$(size(ds["time"]))")
println("Spatialgrid:$(size(ds["temperature"])[1:2])")
println("Totaldatapoints:$(prod(size(ds["temperature"])))")
end

#Considerloadingsubsetsforverylargefiles
```

###InterpolationPerformance

-**Nearest-neighborinterpolation**:Fast,suitableforsimilargridresolutions
-**Bilinearinterpolation**:Moreaccuratebutslower(plannedenhancement)
-**Gridmatching**:BestperformancewhenNetCDFgridmatchessimulationgrid

###Caching

Boundarydataiscachedinmemoryafterloading.Formultiplesimulations:

```julia
#Loadonce,reusemultipletimes
temp_boundaries=load_temperature_boundaries("cmb_temp.nc","surface_temp.nc")

#Useinmultiplesimulations
forsiminsimulations
apply_netcdf_temperature_boundaries!(sim.temp_field,temp_boundaries)
end
```

##Troubleshooting

###CommonIssues

####FileNotFound
```
ERROR:NetCDFfilenotfound:cmb_temp.nc
```
**Solution**:Ensurefilepathsarecorrectandfilesexist.

####GridMismatch
```
ERROR:Innerboundarynlat(32)!=confignlat(64)
```
**Solution**:Eithercreatematchinggridsorrelyonautomaticinterpolation.

####MissingCoordinates
```
WARNING:Nocoordinatesinboundarydataandsizemismatch
```
**Solution**:AddcoordinatevariablestoNetCDFfileorensuredatadimensionsmatchexactly.

####TimeIndexError
```
ERROR:time_index=15outofrange[1,10]
```
**Solution**:Checktimearraylengthandadjusttimeindexing.

###Debugging

Enabledetailedlogging:

```julia
#Checkboundarydatadetails
boundary_stats=get_boundary_statistics(boundary_data)
for(key,val)inboundary_stats
println("$key:$val")
end

#Verifycoordinatearrays
ifboundary_data.theta!==nothing
println("Thetarange:[$(minimum(boundary_data.theta)),$(maximum(boundary_data.theta))]")
end

ifboundary_data.phi!==nothing
println("Phirange:[$(minimum(boundary_data.phi)),$(maximum(boundary_data.phi))]")
end
```

##ExamplesandApplications

###RealisticEarth-likeBoundaries

```julia
#CMBtemperaturewithplumestructures
temp_boundaries=load_temperature_boundaries("realistic_cmb.nc","earth_surface.nc")

#Applywithrealisticsimulationparameters
config=create_optimized_config(128,128,nlat=256,nlon=512)#Highresolution
domain=create_radial_domain(0.35,1.0,128)#Earth-likeradii

temp_field=create_shtns_temperature_field(Float64,config,domain)
apply_netcdf_temperature_boundaries!(temp_field,temp_boundaries)
```

###Time-VaryingSurfaceConditions

```julia
#Seasonalorgeologicaltime-scalesurfacevariations
surface_boundaries=load_temperature_boundaries("cmb_steady.nc","surface_varying.nc")

#Updateduringsimulation
fortimestepin1:simulation_steps
geological_time=timestep*dt*1e6#Converttoyears
update_temperature_boundaries_from_netcdf!(temp_field,surface_boundaries,timestep,dt)

#Continuesimulation...
end
```

###Multi-ComponentComposition

```julia
#LightelementreleaseatCMB
light_element_bc=load_composition_boundaries("cmb_light_elements.nc","surface_zero.nc")
apply_netcdf_composition_boundaries!(comp_field,light_element_bc)

#Additionalcompositionaltracer
heavy_element_bc=load_composition_boundaries("cmb_heavy.nc","surface_heavy.nc")
#Applytosecondcompositionfield...
```

##APIReference

###MainFunctions

-`load_temperature_boundaries(inner_file,outer_file;precision=Float64)`:Loadtemperatureboundaries
-`load_composition_boundaries(inner_file,outer_file;precision=Float64)`:Loadcompositionboundaries
-`apply_netcdf_temperature_boundaries!(field,boundaries,time=0.0)`:Applytemperatureboundaries
-`apply_netcdf_composition_boundaries!(field,boundaries,time=0.0)`:Applycompositionboundaries

###UtilityFunctions

-`read_netcdf_boundary_data(file,field="",coord_names=default_coord_names())`:Low-levelNetCDFreader
-`interpolate_boundary_to_grid(boundary_data,target_theta,target_phi,time_idx=1)`:Manualinterpolation
-`get_boundary_statistics(boundary_data)`:Statisticalanalysis
-`print_boundary_info(boundary_set)`:Displayboundaryinformation
-`validate_netcdf_temperature_compatibility(boundaries,config)`:Validatetemperatureboundaries
-`validate_netcdf_composition_compatibility(boundaries,config)`:Validatecompositionboundaries

###DataStructures

-`BoundaryData{T}`:Singleboundarydata(innerorouter)
-`BoundaryConditionSet{T}`:Completesetwithinnerandouterboundaries

ForcompleteAPIdocumentation,seethefunctiondocstringsinthesourcecode.