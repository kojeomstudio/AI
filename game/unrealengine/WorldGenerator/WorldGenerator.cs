// WorldGenerator.cs
// Phase 18 — Polygon-based heightmap generation with configurable smoothing.
//
// Changes from Phase 17-3:
//   - Reads WorldGenConfig.json for all tunable parameters (smoothing, height profile, noise).
//   - Redesigned height profile: wider gradients for Civilization-style gentle terrain.
//   - Multi-pass box blur smoothing eliminates spiky "needle" terrain.
//   - Re-enforces port/strait features after smoothing to preserve navigability.
//   - Reduced Perlin noise amplitude for smoother land surface.
//
// Usage:
//   dotnet run -- [repo-root-path]
//
// Output dimensions: 8065 x 2017 vertices
//   = 64 x 16 UE Landscape components, 2 subsections x 63 quads each.
//
// Pipeline:
//   1. Load config (WorldGenConfig.json) + data (Coastlines.json, OceanZones.json)
//   2. Polygon land mask (ray-casting point-in-polygon per landmass)
//   3. Port anchor pass (force harbour water + coastal ring)
//   4. Strait enforcement (carve minimum-width channels)
//   5. BFS coast distance (signed pixel distance from land/sea boundary)
//   6. Terrain shaping (distance-based height profile from config)
//   7. Land Perlin noise (configurable octaves + amplitude)
//   8. Landmark height boosts (cape/mountain features)
//   9. Strait height clamping (shallow water in channels)
//  10. Port anchor height smoothing (harbour + beach ring)
//  11. Multi-pass box blur smoothing (configurable passes + radius)
//  12. Coastline sign preservation (land>=Shore, sea<=HarbourWater)
//  13. Re-enforce critical features (ports/straits) after smoothing
//  14. Final [0,1] clamp
//
// Outputs:
//   GameData/WorldHeightmap.r16  — 16-bit unsigned RAW, little-endian
//   GameData/ZoneMask.png        — per-zone colour overlay (debug)
//   GameData/LandMask.png        — binary land/sea mask (debug)

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// ========================================================================
// Data models: OceanZones.json
// ========================================================================

class WorldProjection
{
    [JsonPropertyName("OriginLon")]      public double OriginLon      { get; set; }
    [JsonPropertyName("OriginLat")]      public double OriginLat      { get; set; }
    [JsonPropertyName("UnitsPerDegree")] public double UnitsPerDegree { get; set; } = 7000.0;
}

class OceanZone
{
    [JsonPropertyName("ZoneId")]  public string ZoneId  { get; set; } = "";
    [JsonPropertyName("LonMin")]  public double LonMin  { get; set; }
    [JsonPropertyName("LonMax")]  public double LonMax  { get; set; }
    [JsonPropertyName("LatMin")]  public double LatMin  { get; set; }
    [JsonPropertyName("LatMax")]  public double LatMax  { get; set; }
}

class OceanZonesConfig
{
    [JsonPropertyName("WorldProjection")] public WorldProjection WorldProjection { get; set; } = new();
    [JsonPropertyName("Zones")]           public OceanZone[]     Zones           { get; set; } = [];
}

// ========================================================================
// Data models: Coastlines.json
// ========================================================================

class Landmass
{
    [JsonPropertyName("Id")]         public string     Id         { get; set; } = "";
    [JsonPropertyName("Priority")]   public int        Priority   { get; set; }
    [JsonPropertyName("HeightBase")] public float      HeightBase { get; set; } = 0.62f;
    [JsonPropertyName("Polygon")]    public double[][] Polygon    { get; set; } = [];
}

class StraitDef
{
    [JsonPropertyName("Id")]             public string Id             { get; set; } = "";
    [JsonPropertyName("LonCenter")]      public double LonCenter      { get; set; }
    [JsonPropertyName("LatCenter")]      public double LatCenter      { get; set; }
    [JsonPropertyName("WidthDeg")]       public double WidthDeg       { get; set; }
    [JsonPropertyName("Orientation")]    public int    Orientation    { get; set; } = 90;
    [JsonPropertyName("MinWidthPixels")] public int    MinWidthPixels { get; set; } = 2;
}

class LandmarkDef
{
    [JsonPropertyName("Id")]          public string Id          { get; set; } = "";
    [JsonPropertyName("Lon")]         public double Lon         { get; set; }
    [JsonPropertyName("Lat")]         public double Lat         { get; set; }
    [JsonPropertyName("Type")]        public string Type        { get; set; } = "";
    [JsonPropertyName("HeightBoost")] public float  HeightBoost { get; set; }
    [JsonPropertyName("RadiusDeg")]   public double RadiusDeg   { get; set; }
}

class PortAnchor
{
    [JsonPropertyName("PortId")] public string PortId { get; set; } = "";
    [JsonPropertyName("Lon")]    public double Lon    { get; set; }
    [JsonPropertyName("Lat")]    public double Lat    { get; set; }
}

class CoastlinesConfig
{
    [JsonPropertyName("Landmasses")]  public Landmass[]    Landmasses  { get; set; } = [];
    [JsonPropertyName("Straits")]     public StraitDef[]   Straits     { get; set; } = [];
    [JsonPropertyName("Landmarks")]   public LandmarkDef[] Landmarks   { get; set; } = [];
    [JsonPropertyName("PortAnchors")] public PortAnchor[]  PortAnchors { get; set; } = [];
}

// ========================================================================
// Data models: WorldGenConfig.json
// ========================================================================

class HeightmapConfig
{
    [JsonPropertyName("Width")]              public int   Width              { get; set; } = 8065;
    [JsonPropertyName("Height")]             public int   Height             { get; set; } = 2017;
    [JsonPropertyName("SmoothingPasses")]    public int   SmoothingPasses    { get; set; } = 4;
    [JsonPropertyName("SmoothingRadius")]    public int   SmoothingRadius    { get; set; } = 4;
    [JsonPropertyName("PerlinOctaves")]      public int   PerlinOctaves      { get; set; } = 4;
    [JsonPropertyName("PerlinPersistence")]  public float PerlinPersistence  { get; set; } = 0.5f;
    [JsonPropertyName("PerlinBaseFrequency")]public float PerlinBaseFrequency{ get; set; } = 0.003125f;
    [JsonPropertyName("LandNoiseAmplitude")] public float LandNoiseAmplitude { get; set; } = 0.04f;
}

class HeightDistancesConfig
{
    [JsonPropertyName("ShelfWidth")]       public int ShelfWidth       { get; set; } = 8;
    [JsonPropertyName("SlopeWidth")]       public int SlopeWidth       { get; set; } = 25;
    [JsonPropertyName("AbyssalApproach")]  public int AbyssalApproach  { get; set; } = 60;
    [JsonPropertyName("CoastalLowland")]   public int CoastalLowland   { get; set; } = 25;
    [JsonPropertyName("InlandHills")]      public int InlandHills      { get; set; } = 80;
    [JsonPropertyName("InteriorPlateau")]  public int InteriorPlateau  { get; set; } = 150;
}

class HeightProfileConfig
{
    [JsonPropertyName("ReferenceHeightBase")] public float ReferenceHeightBase { get; set; } = 0.62f;
    [JsonPropertyName("DeepOcean")]        public float DeepOcean        { get; set; } = 0.10f;
    [JsonPropertyName("AbyssalApproach")]  public float AbyssalApproach  { get; set; } = 0.20f;
    [JsonPropertyName("ContinentalSlope")] public float ContinentalSlope { get; set; } = 0.35f;
    [JsonPropertyName("ShallowShelf")]     public float ShallowShelf     { get; set; } = 0.45f;
    [JsonPropertyName("ShallowStrait")]    public float ShallowStrait    { get; set; } = 0.40f;
    [JsonPropertyName("HarbourWater")]     public float HarbourWater     { get; set; } = 0.48f;
    [JsonPropertyName("Shore")]            public float Shore            { get; set; } = 0.53f;
    [JsonPropertyName("CoastalLowland")]   public float CoastalLowland   { get; set; } = 0.55f;
    [JsonPropertyName("InlandHills")]      public float InlandHills      { get; set; } = 0.57f;
    [JsonPropertyName("InteriorPlateau")]  public float InteriorPlateau  { get; set; } = 0.58f;
    [JsonPropertyName("Distances")]        public HeightDistancesConfig Distances { get; set; } = new();
}

class WorldExtentConfig
{
    [JsonPropertyName("LonMin")] public double LonMin { get; set; } = -180.0;
    [JsonPropertyName("LonMax")] public double LonMax { get; set; } =  180.0;
    [JsonPropertyName("LatMin")] public double LatMin { get; set; } =  -70.0;
    [JsonPropertyName("LatMax")] public double LatMax { get; set; } =   80.0;
}

class PortsConfig
{
    [JsonPropertyName("PlacementEnabled")]  public bool   PlacementEnabled  { get; set; } = true;
    [JsonPropertyName("CoastlinesJsonPath")]public string CoastlinesJsonPath{ get; set; } = "GameData/Coastlines.json";
    [JsonPropertyName("AnchorRadius")]      public int    AnchorRadius      { get; set; } = 3;
}

class StraitsConfig
{
    [JsonPropertyName("ChannelHalfLength")] public int ChannelHalfLength { get; set; } = 20;
}

class WorldGenConfig
{
    [JsonPropertyName("WorldExtent")]  public WorldExtentConfig  WorldExtent  { get; set; } = new();
    [JsonPropertyName("Heightmap")]    public HeightmapConfig    Heightmap    { get; set; } = new();
    [JsonPropertyName("HeightProfile")]public HeightProfileConfig HeightProfile{ get; set; } = new();
    [JsonPropertyName("Ports")]        public PortsConfig        Ports        { get; set; } = new();
    [JsonPropertyName("Straits")]      public StraitsConfig      Straits      { get; set; } = new();
}

// ========================================================================
// Main program
// ========================================================================

class Program
{
    // Map dimensions — UE Landscape compatible: 64 x 16 components, 2 subsections x 63 quads.
    static int MAP_W = 8065;
    static int MAP_H = 2017;

    // Game world lon/lat extent (read from WorldGenConfig.json "WorldExtent" section).
    static double LON_MIN = -180.0;
    static double LON_MAX =  180.0;
    static double LAT_MIN =  -70.0;
    static double LAT_MAX =   80.0;

    // Pipeline configuration (loaded from WorldGenConfig.json or defaults).
    static WorldGenConfig Config = new();

    // Zone colours for ZoneMask.png
    static readonly (byte R, byte G, byte B)[] ZoneColors =
    [
        ( 0, 100, 180),  // REGION_MED
        (30,  60, 160),  // REGION_NATL
        (50,  80, 200),  // REGION_SATL
        (70, 150, 200),  // REGION_IND
        ( 0, 180, 160),  // (reserved)
        (140, 200, 230), // REGION_NORTH_SEA
        (200, 240, 255), // REGION_ARCTIC
        ( 0,  40, 180),  // REGION_PACIFIC
        (180, 130,  60), // REGION_WEST_AFRICA
        (120, 170,  80), // REGION_EAST_AFRICA
    ];

    // ========================================================================
    // Entry point
    // ========================================================================

    static void Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

        string exeDir   = AppContext.BaseDirectory;
        string repoRoot = args.Length > 0
            ? Path.GetFullPath(args[0])
            : Path.GetFullPath(Path.Combine(exeDir, "..", "..", "..", "..", ".."));

        // -- Step 1: Load configuration ------------------------------------------
        string configPath = Path.Combine(repoRoot, "GameData", "WorldGenConfig.json");
        if (File.Exists(configPath))
        {
            try
            {
                var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                Config = JsonSerializer.Deserialize<WorldGenConfig>(File.ReadAllText(configPath), opts)
                         ?? new WorldGenConfig();
                Console.WriteLine($"[WorldGenerator] Config loaded from {configPath}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[WorldGenerator] WARN: Failed to parse WorldGenConfig.json ({ex.Message}). Using defaults.");
                Config = new WorldGenConfig();
            }
        }
        else
        {
            Console.WriteLine($"[WorldGenerator] WorldGenConfig.json not found at {configPath}. Using defaults.");
        }

        // Apply config dimensions and world extent.
        MAP_W = Config.Heightmap.Width;
        MAP_H = Config.Heightmap.Height;
        LON_MIN = Config.WorldExtent.LonMin;
        LON_MAX = Config.WorldExtent.LonMax;
        LAT_MIN = Config.WorldExtent.LatMin;
        LAT_MAX = Config.WorldExtent.LatMax;

        Console.WriteLine($"[WorldGenerator] Map dimensions: {MAP_W} x {MAP_H}");
        Console.WriteLine($"[WorldGenerator] Smoothing: {Config.Heightmap.SmoothingPasses} passes, radius {Config.Heightmap.SmoothingRadius}");
        Console.WriteLine($"[WorldGenerator] Noise: {Config.Heightmap.PerlinOctaves} octaves, amplitude {Config.Heightmap.LandNoiseAmplitude}");

        // -- Step 2: Load data files ---------------------------------------------
        string zonesPath     = Path.Combine(repoRoot, "GameData", "OceanZones.json");
        string coastPath     = Path.Combine(repoRoot, "GameData", "Coastlines.json");
        string heightmapPath = Path.Combine(repoRoot, "GameData", "WorldHeightmap.r16");
        string zoneMaskPath  = Path.Combine(repoRoot, "GameData", "ZoneMask.png");
        string landMaskPath  = Path.Combine(repoRoot, "GameData", "LandMask.png");

        if (!File.Exists(zonesPath))
        {
            Console.Error.WriteLine($"[WorldGenerator] ERROR: OceanZones.json not found at {zonesPath}");
            Environment.Exit(1);
        }
        if (!File.Exists(coastPath))
        {
            Console.Error.WriteLine($"[WorldGenerator] ERROR: Coastlines.json not found at {coastPath}");
            Environment.Exit(1);
        }

        var opts2   = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var zones   = JsonSerializer.Deserialize<OceanZonesConfig>(File.ReadAllText(zonesPath), opts2)!;
        var coasts  = JsonSerializer.Deserialize<CoastlinesConfig>(File.ReadAllText(coastPath), opts2)!;

        Console.WriteLine($"[WorldGenerator] Loaded {zones.Zones.Length} ocean zones, " +
                          $"{coasts.Landmasses.Length} landmasses, " +
                          $"{coasts.Landmarks.Length} landmarks, " +
                          $"{coasts.PortAnchors.Length} port anchors.");
        Console.WriteLine($"[WorldGenerator] Building {MAP_W} x {MAP_H} heightmap...");

        // Sort landmasses by priority ascending so highest-priority overwrites last.
        var sortedLandmasses = coasts.Landmasses;
        Array.Sort(sortedLandmasses, (a, b) => a.Priority.CompareTo(b.Priority));

        // -- Steps 2-4: Build land mask ------------------------------------------
        Console.WriteLine("[WorldGenerator] [Step 2-4] Building land mask (polygons + ports + straits)...");
        int[] landOwner;
        bool[] isLand = BuildLandMask(sortedLandmasses, coasts.PortAnchors, coasts.Straits, out landOwner);

        // -- Step 5: BFS coast distance ------------------------------------------
        Console.WriteLine("[WorldGenerator] [Step 5] Computing coast distance (BFS)...");
        int[] coastDist = ComputeCoastDistance(isLand);

        // -- Steps 6-10: Build heightmap -----------------------------------------
        Console.WriteLine("[WorldGenerator] [Step 6-10] Building heightmap (terrain shaping)...");
        float[] hmap = BuildHeightmap(isLand, coastDist, landOwner, sortedLandmasses,
                                      coasts.Landmarks, coasts.Straits, coasts.PortAnchors, zones.Zones);

        // -- Step 11: Multi-pass smoothing ---------------------------------------
        if (Config.Heightmap.SmoothingPasses > 0)
        {
            Console.WriteLine($"[WorldGenerator] [Step 11] Smoothing heightmap ({Config.Heightmap.SmoothingPasses} passes, radius {Config.Heightmap.SmoothingRadius})...");
            SmoothHeightmap(hmap, Config.Heightmap.SmoothingPasses, Config.Heightmap.SmoothingRadius);
        }

        // -- Step 12: Coastline sign preservation after smoothing ----------------
        // Smoothing blurs values across the land/sea boundary. Pixels that were
        // originally land (isLand=true) might drop below 0.50, and sea pixels
        // might rise above 0.50. This breaks the coastline alignment with the
        // Water Plugin (water surface = Z=0 = height 0.50).
        //
        // Fix: clamp each pixel so its height respects its original land/sea
        // designation from the polygon mask. Land pixels stay >= Shore (0.52),
        // sea pixels stay <= HarbourWater (0.49). This preserves the smooth
        // shapes from blurring while guaranteeing the coastline boundary.
        {
            var hp = Config.HeightProfile;
            int fixedLand = 0, fixedSea = 0;
            for (int i = 0; i < hmap.Length; i++)
            {
                if (isLand[i])
                {
                    // Land pixel must stay above sea level.
                    if (hmap[i] < hp.Shore)
                    {
                        hmap[i] = hp.Shore;
                        fixedLand++;
                    }
                }
                else
                {
                    // Sea pixel must stay below sea level.
                    if (hmap[i] > hp.HarbourWater)
                    {
                        hmap[i] = hp.HarbourWater;
                        fixedSea++;
                    }
                }
            }
            Console.WriteLine($"[WorldGenerator] [Step 12] Coastline sign preservation: {fixedLand} land pixels lifted, {fixedSea} sea pixels clamped.");
        }

        // -- Step 13: Re-enforce critical features after smoothing ---------------
        // Smoothing blurs port harbours and strait channels. Restore them to
        // guarantee navigability.
        Console.WriteLine("[WorldGenerator] [Step 13] Re-enforcing ports and straits after smoothing...");
        ReEnforcePorts(hmap, coasts.PortAnchors);
        ReEnforceStraits(hmap, coasts.Straits);

        // -- Step 14: Final clamp ------------------------------------------------
        for (int i = 0; i < hmap.Length; i++)
            hmap[i] = Math.Clamp(hmap[i], 0.0f, 1.0f);

        // -- Write splatmap & colormap -------------------------------------------
        string splatmapPath = Path.Combine(repoRoot, "GameData", "TerrainSplatmap.png");
        string colormapPath = Path.Combine(repoRoot, "GameData", "TerrainColormap.png");
        WriteSplatmap(hmap, isLand, coastDist, splatmapPath);
        Console.WriteLine($"[WorldGenerator] Wrote {splatmapPath}");
        WriteColormap(hmap, isLand, coastDist, colormapPath);
        Console.WriteLine($"[WorldGenerator] Wrote {colormapPath}");

        // -- Write outputs -------------------------------------------------------
        WriteR16(hmap, heightmapPath);
        Console.WriteLine($"[WorldGenerator] Wrote {heightmapPath}  ({MAP_W * MAP_H * 2 / 1024} KB)");

        WriteZoneMask(zones.Zones, zoneMaskPath);
        Console.WriteLine($"[WorldGenerator] Wrote {zoneMaskPath}");

        WriteLandMask(isLand, landMaskPath);
        Console.WriteLine($"[WorldGenerator] Wrote {landMaskPath}");

        // -- Port diagnostics ----------------------------------------------------
        foreach (var anchor in coasts.PortAnchors)
        {
            var (px, py) = LonLatToPixel(anchor.Lon, anchor.Lat);
            int idx = py * MAP_W + px;
            float h = hmap[idx];
            int cd  = coastDist[idx];
            string status = h < 0.50f ? "water" : "land";
            string coastal = (cd >= -3 && cd <= 3) ? "coastal OK"
                           : cd > 3                ? $"inland (dist={cd})"
                           : $"open-sea (dist={cd})";
            Console.WriteLine($"[WorldGenerator] Port {anchor.PortId}: ({anchor.Lon:F2},{anchor.Lat:F2}) -> pixel({px},{py}) h={h:F4} {status} {coastal}");
        }

        Console.WriteLine("[WorldGenerator] Done.");
    }

    // ========================================================================
    // Coordinate helpers
    // ========================================================================

    static (double Lon, double Lat) PixelToLonLat(int px, int py)
    {
        double lon = LON_MIN + (double)px / (MAP_W - 1) * (LON_MAX - LON_MIN);
        double lat = LAT_MAX - (double)py / (MAP_H - 1) * (LAT_MAX - LAT_MIN);
        return (lon, lat);
    }

    static (int Px, int Py) LonLatToPixel(double lon, double lat)
    {
        int px = (int)Math.Round((lon - LON_MIN) / (LON_MAX - LON_MIN) * (MAP_W - 1));
        int py = (int)Math.Round((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (MAP_H - 1));
        px = Math.Clamp(px, 0, MAP_W - 1);
        py = Math.Clamp(py, 0, MAP_H - 1);
        return (px, py);
    }

    // ========================================================================
    // Steps 2-4: Land mask (ray-casting point-in-polygon)
    // ========================================================================

    static bool[] BuildLandMask(Landmass[] landmasses, PortAnchor[] portAnchors, StraitDef[] straits,
                                out int[] landOwner)
    {
        bool[] isLand = new bool[MAP_W * MAP_H];
        landOwner = new int[MAP_W * MAP_H];
        Array.Fill(landOwner, -1);

        // Step 2: Polygon fill — mark pixels inside any landmass polygon.
        // Track which landmass owns each pixel (highest priority wins).
        for (int py = 0; py < MAP_H; py++)
        {
            for (int px = 0; px < MAP_W; px++)
            {
                (double lon, double lat) = PixelToLonLat(px, py);
                int idx = py * MAP_W + px;
                int bestPriority = -1;

                for (int li = 0; li < landmasses.Length; li++)
                {
                    Landmass lm = landmasses[li];
                    if (IsPointInPolygon(lon, lat, lm.Polygon))
                    {
                        isLand[idx] = true;
                        if (lm.Priority > bestPriority)
                        {
                            bestPriority = lm.Priority;
                            landOwner[idx] = li;
                        }
                    }
                }
            }

            if (py % 200 == 0)
                Console.Write($"\r[WorldGenerator]   Land mask: {py * 100 / MAP_H}%   ");
        }
        Console.WriteLine($"\r[WorldGenerator]   Land mask: 100%   ");

        // Step 3: Port anchor pass — guarantee each port pixel is coastal water.
        // Force port pixel to sea, create land ring around it.
        int AnchorRadius = Config.Ports.AnchorRadius;
        foreach (var anchor in portAnchors)
        {
            var (ax, ay) = LonLatToPixel(anchor.Lon, anchor.Lat);

            // Force port pixel to sea (harbour water).
            if (ax >= 0 && ax < MAP_W && ay >= 0 && ay < MAP_H)
                isLand[ay * MAP_W + ax] = false;

            // Mark outer ring as land (creates coastal beach around the harbour).
            for (int dy = -AnchorRadius; dy <= AnchorRadius; dy++)
            {
                for (int dx = -AnchorRadius; dx <= AnchorRadius; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = ax + dx, ny = ay + dy;
                    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
                    int distSq = dx * dx + dy * dy;
                    if (distSq >= (AnchorRadius - 1) * (AnchorRadius - 1) && distSq <= AnchorRadius * AnchorRadius)
                        isLand[ny * MAP_W + nx] = true;
                }
            }
        }

        // Step 4: Strait enforcement — carve minimum-width sea channels.
        // Orientation=90 = E-W channel (horizontal), Orientation=0 = N-S channel.
        int HalfLen = Config.Straits.ChannelHalfLength;
        foreach (var strait in straits)
        {
            var (scx, scy) = LonLatToPixel(strait.LonCenter, strait.LatCenter);
            int halfW = Math.Max(1, strait.MinWidthPixels / 2);

            if (strait.Orientation == 90)
            {
                for (int px = scx - HalfLen; px <= scx + HalfLen; px++)
                {
                    if (px < 0 || px >= MAP_W) continue;
                    for (int dy = -halfW; dy <= halfW; dy++)
                    {
                        int ny = scy + dy;
                        if (ny < 0 || ny >= MAP_H) continue;
                        isLand[ny * MAP_W + px] = false;
                    }
                }
            }
            else
            {
                for (int py = scy - HalfLen; py <= scy + HalfLen; py++)
                {
                    if (py < 0 || py >= MAP_H) continue;
                    for (int dx = -halfW; dx <= halfW; dx++)
                    {
                        int nx = scx + dx;
                        if (nx < 0 || nx >= MAP_W) continue;
                        isLand[py * MAP_W + nx] = false;
                    }
                }
            }

            Console.WriteLine($"[WorldGenerator]   Strait {strait.Id}: carved {strait.MinWidthPixels}px channel at pixel({scx},{scy})");
        }

        return isLand;
    }

    // Ray-casting point-in-polygon (Jordan curve theorem).
    static bool IsPointInPolygon(double lon, double lat, double[][] poly)
    {
        int n = poly.Length;
        if (n < 3) return false;

        bool inside = false;
        int j = n - 1;
        for (int i = 0; i < n; i++)
        {
            double xi = poly[i][0], yi = poly[i][1];
            double xj = poly[j][0], yj = poly[j][1];

            bool crosses = ((yi > lat) != (yj > lat)) &&
                           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi);
            if (crosses) inside = !inside;
            j = i;
        }
        return inside;
    }

    // ========================================================================
    // Step 5: BFS coast distance
    // ========================================================================

    // Returns per-pixel signed distance to the nearest land/sea boundary.
    // Land pixels: positive distance (inland depth).
    // Sea pixels: negative distance (offshore depth).
    // Cap at +/-128.
    static int[] ComputeCoastDistance(bool[] isLand)
    {
        int[] dist = new int[MAP_W * MAP_H];
        Array.Fill(dist, int.MaxValue);

        var queue = new Queue<int>();
        int[] dx4 = { -1, 1, 0, 0 };
        int[] dy4 = {  0, 0,-1, 1 };

        // Seed: boundary pixels (land adjacent to sea or vice versa).
        for (int py = 0; py < MAP_H; py++)
        {
            for (int px = 0; px < MAP_W; px++)
            {
                int idx = py * MAP_W + px;
                bool land = isLand[idx];
                bool boundary = false;

                for (int d = 0; d < 4; d++)
                {
                    int nx = px + dx4[d], ny = py + dy4[d];
                    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
                    if (isLand[ny * MAP_W + nx] != land) { boundary = true; break; }
                }

                if (boundary)
                {
                    dist[idx] = 0;
                    queue.Enqueue(idx);
                }
            }
        }

        // BFS flood fill from boundary outward.
        while (queue.Count > 0)
        {
            int idx = queue.Dequeue();
            int px  = idx % MAP_W;
            int py  = idx / MAP_W;
            int d   = dist[idx];

            for (int dir = 0; dir < 4; dir++)
            {
                int nx = px + dx4[dir], ny = py + dy4[dir];
                if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
                int nidx = ny * MAP_W + nx;
                if (dist[nidx] == int.MaxValue)
                {
                    dist[nidx] = d + 1;
                    queue.Enqueue(nidx);
                }
            }
        }

        // Sign: land = positive, sea = negative. Cap at +/-128.
        for (int i = 0; i < dist.Length; i++)
        {
            int s = isLand[i] ? dist[i] : -dist[i];
            dist[i] = Math.Clamp(s, -128, 128);
        }

        return dist;
    }

    // ========================================================================
    // Steps 6-10: Build heightmap
    // ========================================================================

    static float[] BuildHeightmap(bool[] isLand, int[] coastDist, int[] landOwner,
                                   Landmass[] landmasses, LandmarkDef[] landmarks,
                                   StraitDef[] straits, PortAnchor[] portAnchors, OceanZone[] zones)
    {
        HeightProfileConfig hp = Config.HeightProfile;
        HeightDistancesConfig hd = hp.Distances;
        float[] hmap = new float[MAP_W * MAP_H];

        // Reference HeightBase used to normalize per-landmass scaling.
        // Landmasses with HeightBase > reference get taller plateaus, smaller get shorter.
        float ReferenceHeightBase = hp.ReferenceHeightBase;

        // Step 6: Ocean floor baseline + distance-based terrain shaping.
        // Phase 19: Uses per-landmass HeightBase to scale interior plateau height.
        // Sea-side: gradual slope from deep ocean to continental shelf.
        // Land-side: gentle rise from shore to interior plateau (scaled by HeightBase).
        for (int i = 0; i < hmap.Length; i++)
        {
            int d = coastDist[i];

            if (d < 0)
            {
                // --- Sea pixel (d is negative, magnitude = distance from coast) ---
                int seaDist = -d;

                if (seaDist <= hd.ShelfWidth)
                {
                    // Continental shelf: shallow water near coast.
                    float t = (float)seaDist / hd.ShelfWidth;
                    hmap[i] = Lerp(hp.ShallowShelf, hp.ContinentalSlope, t);
                }
                else if (seaDist <= hd.SlopeWidth)
                {
                    // Continental slope: moderate descent.
                    float t = (float)(seaDist - hd.ShelfWidth) / (hd.SlopeWidth - hd.ShelfWidth);
                    hmap[i] = Lerp(hp.ContinentalSlope, hp.AbyssalApproach, t);
                }
                else if (seaDist <= hd.AbyssalApproach)
                {
                    // Abyssal approach: gradual descent to deep ocean floor.
                    float t = (float)(seaDist - hd.SlopeWidth) / (hd.AbyssalApproach - hd.SlopeWidth);
                    hmap[i] = Lerp(hp.AbyssalApproach, hp.DeepOcean, t);
                }
                else
                {
                    // Deep ocean floor: flat.
                    hmap[i] = hp.DeepOcean;
                }
            }
            else
            {
                // --- Land pixel (d is positive, magnitude = distance from coast) ---
                // Compute effective plateau height scaled by landmass HeightBase.
                float heightBaseScale = 1.0f;
                if (landOwner[i] >= 0 && landOwner[i] < landmasses.Length)
                {
                    float landHeightBase = landmasses[landOwner[i]].HeightBase;
                    heightBaseScale = landHeightBase / ReferenceHeightBase;
                }
                float effectivePlateau = hp.InteriorPlateau * heightBaseScale;
                float effectiveInlandHills = hp.InlandHills * heightBaseScale;

                if (d == 0)
                {
                    // Shore: just above sea level.
                    hmap[i] = hp.Shore;
                }
                else if (d <= hd.CoastalLowland)
                {
                    // Coastal lowland: gentle rise from shore.
                    float t = (float)d / hd.CoastalLowland;
                    hmap[i] = Lerp(hp.Shore, hp.CoastalLowland, t);
                }
                else if (d <= hd.InlandHills)
                {
                    // Inland hills: rolling terrain (scaled by HeightBase).
                    float t = (float)(d - hd.CoastalLowland) / (hd.InlandHills - hd.CoastalLowland);
                    hmap[i] = Lerp(hp.CoastalLowland, effectiveInlandHills, t);
                }
                else if (d <= hd.InteriorPlateau)
                {
                    // Interior plateau approach (scaled by HeightBase).
                    float t = (float)(d - hd.InlandHills) / (hd.InteriorPlateau - hd.InlandHills);
                    hmap[i] = Lerp(effectiveInlandHills, effectivePlateau, t);
                }
                else
                {
                    // Deep interior plateau (scaled by HeightBase).
                    hmap[i] = effectivePlateau;
                }
            }
        }

        // Step 7: Land-only Perlin noise.
        float noiseAmp = Config.Heightmap.LandNoiseAmplitude;
        if (noiseAmp > 0f)
        {
            float[] noise = new float[MAP_W * MAP_H];
            AddPerlinNoise(noise,
                octaves: Config.Heightmap.PerlinOctaves,
                persistence: Config.Heightmap.PerlinPersistence,
                baseFrequency: Config.Heightmap.PerlinBaseFrequency,
                seed: 42);
            for (int i = 0; i < hmap.Length; i++)
            {
                if (isLand[i])
                    hmap[i] += noise[i] * noiseAmp;
            }
        }

        // Step 8: Landmark cape height boosts.
        foreach (var lm in landmarks)
        {
            if (lm.HeightBoost <= 0f) continue;

            var (cx, cy) = LonLatToPixel(lm.Lon, lm.Lat);
            double radiusPx = lm.RadiusDeg / (LON_MAX - LON_MIN) * (MAP_W - 1);

            for (int dy = -(int)(radiusPx + 1); dy <= (int)(radiusPx + 1); dy++)
            {
                for (int dx = -(int)(radiusPx + 1); dx <= (int)(radiusPx + 1); dx++)
                {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
                    int idx = ny * MAP_W + nx;
                    if (!isLand[idx]) continue;

                    double distSq = (double)(dx * dx + dy * dy);
                    double rSq    = radiusPx * radiusPx;
                    if (distSq > rSq) continue;

                    float blend = 1.0f - (float)(distSq / rSq);
                    hmap[idx] += lm.HeightBoost * blend;
                }
            }
        }

        // Step 9: Strait channel height enforcement.
        // Clamp strait pixels to ShallowStrait to distinguish from deep ocean.
        int SHalfLen = Config.Straits.ChannelHalfLength;
        foreach (var strait in straits)
        {
            var (scx, scy) = LonLatToPixel(strait.LonCenter, strait.LatCenter);
            int halfW = Math.Max(1, strait.MinWidthPixels / 2);

            if (strait.Orientation == 90)
            {
                for (int px = scx - SHalfLen; px <= scx + SHalfLen; px++)
                {
                    if (px < 0 || px >= MAP_W) continue;
                    for (int dy = -halfW; dy <= halfW; dy++)
                    {
                        int ny = scy + dy;
                        if (ny < 0 || ny >= MAP_H) continue;
                        hmap[ny * MAP_W + px] = Math.Min(hmap[ny * MAP_W + px], hp.ShallowStrait);
                    }
                }
            }
            else
            {
                for (int py = scy - SHalfLen; py <= scy + SHalfLen; py++)
                {
                    if (py < 0 || py >= MAP_H) continue;
                    for (int dx = -halfW; dx <= halfW; dx++)
                    {
                        int nx = scx + dx;
                        if (nx < 0 || nx >= MAP_W) continue;
                        hmap[py * MAP_W + nx] = Math.Min(hmap[py * MAP_W + nx], hp.ShallowStrait);
                    }
                }
            }
        }

        // Step 10: Port anchor smoothing — ensure port + ring are harbour-depth.
        foreach (var anchor in portAnchors)
        {
            var (ax, ay) = LonLatToPixel(anchor.Lon, anchor.Lat);

            if (ax >= 0 && ax < MAP_W && ay >= 0 && ay < MAP_H)
                hmap[ay * MAP_W + ax] = hp.HarbourWater;

            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = ax + dx, ny = ay + dy;
                    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
                    int idx = ny * MAP_W + nx;
                    if (isLand[idx])
                        hmap[idx] = Math.Min(hmap[idx], hp.CoastalLowland);
                }
            }
        }

        return hmap;
    }

    static float Lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }

    // ========================================================================
    // Step 11: Multi-pass box blur smoothing
    // ========================================================================

    // Box blur approximates Gaussian blur when applied in multiple passes.
    // 3-4 passes of box blur give a very good Gaussian approximation.
    // This is O(n) per pass regardless of radius (sliding window).
    static void SmoothHeightmap(float[] hmap, int passes, int radius)
    {
        float[] tmp = new float[hmap.Length];

        for (int p = 0; p < passes; p++)
        {
            // Horizontal pass: hmap -> tmp
            BoxBlurHorizontal(hmap, tmp, MAP_W, MAP_H, radius);
            // Vertical pass: tmp -> hmap
            BoxBlurVertical(tmp, hmap, MAP_W, MAP_H, radius);

            if (p % 2 == 0)
                Console.Write($"\r[WorldGenerator]   Smoothing pass {p + 1}/{passes}...");
        }
        Console.WriteLine($"\r[WorldGenerator]   Smoothing: {passes} passes complete.       ");
    }

    static void BoxBlurHorizontal(float[] src, float[] dst, int w, int h, int radius)
    {
        float invKernel = 1.0f / (2 * radius + 1);

        for (int y = 0; y < h; y++)
        {
            int offset = y * w;
            float sum = 0;

            // Initialize sliding window with clamped boundary.
            for (int x = -radius; x <= radius; x++)
            {
                int idx = Math.Clamp(x, 0, w - 1);
                sum += src[offset + idx];
            }
            dst[offset] = sum * invKernel;

            // Slide window across the row.
            for (int x = 1; x < w; x++)
            {
                int addIdx = Math.Clamp(x + radius, 0, w - 1);
                int subIdx = Math.Clamp(x - radius - 1, 0, w - 1);
                sum += src[offset + addIdx] - src[offset + subIdx];
                dst[offset + x] = sum * invKernel;
            }
        }
    }

    static void BoxBlurVertical(float[] src, float[] dst, int w, int h, int radius)
    {
        float invKernel = 1.0f / (2 * radius + 1);

        for (int x = 0; x < w; x++)
        {
            float sum = 0;

            // Initialize sliding window.
            for (int y = -radius; y <= radius; y++)
            {
                int idx = Math.Clamp(y, 0, h - 1) * w + x;
                sum += src[idx];
            }
            dst[x] = sum * invKernel;

            // Slide window down the column.
            for (int y = 1; y < h; y++)
            {
                int addIdx = Math.Clamp(y + radius, 0, h - 1) * w + x;
                int subIdx = Math.Clamp(y - radius - 1, 0, h - 1) * w + x;
                sum += src[addIdx] - src[subIdx];
                dst[y * w + x] = sum * invKernel;
            }
        }
    }

    // ========================================================================
    // Step 12: Re-enforce critical features after smoothing
    // ========================================================================

    // Smoothing blurs harbour depth and strait channels. Restore them so
    // ships can navigate correctly.
    //
    // Port pixel: Math.Min → stays BELOW sea level (harbour water).
    // Port ring:  Math.Max → stays ABOVE sea level (visible land).
    //
    // Bug fix (Phase 18-1): the ring previously used Math.Min which could
    // NOT lift smoothed values back above sea level. Changed to Math.Max
    // so the land ring around each port is always visible above water.
    static void ReEnforcePorts(float[] hmap, PortAnchor[] portAnchors)
    {
        var hp = Config.HeightProfile;

        foreach (var anchor in portAnchors)
        {
            var (ax, ay) = LonLatToPixel(anchor.Lon, anchor.Lat);

            // Port pixel: ensure it stays below sea level (harbour water).
            if (ax >= 0 && ax < MAP_W && ay >= 0 && ay < MAP_H)
            {
                int idx = ay * MAP_W + ax;
                hmap[idx] = Math.Min(hmap[idx], hp.HarbourWater);
            }

            // 1px ring: ensure it stays ABOVE sea level (visible land).
            // Math.Max guarantees the ring is at least Shore height even
            // if smoothing pulled it below 0.50.
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx == 0 && dy == 0) continue;
                    int nx = ax + dx, ny = ay + dy;
                    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) continue;
                    int idx = ny * MAP_W + nx;
                    hmap[idx] = Math.Max(hmap[idx], hp.Shore);
                }
            }
        }
    }

    static void ReEnforceStraits(float[] hmap, StraitDef[] straits)
    {
        var hp = Config.HeightProfile;
        int SHalfLen = Config.Straits.ChannelHalfLength;

        foreach (var strait in straits)
        {
            var (scx, scy) = LonLatToPixel(strait.LonCenter, strait.LatCenter);
            int halfW = Math.Max(1, strait.MinWidthPixels / 2);

            if (strait.Orientation == 90)
            {
                for (int px = scx - SHalfLen; px <= scx + SHalfLen; px++)
                {
                    if (px < 0 || px >= MAP_W) continue;
                    for (int dy = -halfW; dy <= halfW; dy++)
                    {
                        int ny = scy + dy;
                        if (ny < 0 || ny >= MAP_H) continue;
                        hmap[ny * MAP_W + px] = Math.Min(hmap[ny * MAP_W + px], hp.ShallowStrait);
                    }
                }
            }
            else
            {
                for (int py = scy - SHalfLen; py <= scy + SHalfLen; py++)
                {
                    if (py < 0 || py >= MAP_H) continue;
                    for (int dx = -halfW; dx <= halfW; dx++)
                    {
                        int nx = scx + dx;
                        if (nx < 0 || nx >= MAP_W) continue;
                        hmap[py * MAP_W + nx] = Math.Min(hmap[py * MAP_W + nx], hp.ShallowStrait);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Perlin gradient noise
    // ========================================================================

    static void AddPerlinNoise(float[] hmap, int octaves, float persistence,
                               float baseFrequency, int seed)
    {
        const int GW = 512;
        const int GH = 256;
        var gx  = new float[GW * GH];
        var gy  = new float[GW * GH];
        var rng = new Random(seed);

        for (int i = 0; i < GW * GH; i++)
        {
            double angle = rng.NextDouble() * Math.PI * 2.0;
            gx[i] = (float)Math.Cos(angle);
            gy[i] = (float)Math.Sin(angle);
        }

        float freq = baseFrequency;
        float amp  = 0.5f;

        for (int oct = 0; oct < octaves; oct++)
        {
            for (int py = 0; py < MAP_H; py++)
            {
                for (int px = 0; px < MAP_W; px++)
                {
                    float nx = px * freq * GW;
                    float ny = py * freq * GH;

                    int x0 = (int)Math.Floor(nx) & (GW - 1);
                    int y0 = (int)Math.Floor(ny) & (GH - 1);
                    int x1 = (x0 + 1) & (GW - 1);
                    int y1 = (y0 + 1) & (GH - 1);

                    float dx = nx - (float)Math.Floor(nx);
                    float dy = ny - (float)Math.Floor(ny);
                    float sx = dx * dx * (3f - 2f * dx);
                    float sy = dy * dy * (3f - 2f * dy);

                    float n00 = gx[y0 * GW + x0] * dx        + gy[y0 * GW + x0] * dy;
                    float n10 = gx[y0 * GW + x1] * (dx - 1f) + gy[y0 * GW + x1] * dy;
                    float n01 = gx[y1 * GW + x0] * dx        + gy[y1 * GW + x0] * (dy - 1f);
                    float n11 = gx[y1 * GW + x1] * (dx - 1f) + gy[y1 * GW + x1] * (dy - 1f);

                    float ix0 = n00 + sx * (n10 - n00);
                    float ix1 = n01 + sx * (n11 - n01);
                    hmap[py * MAP_W + px] += (ix0 + sy * (ix1 - ix0)) * amp;
                }
            }
            freq *= 2f;
            amp  *= persistence;
        }
    }

    // ========================================================================
    // R16 writer
    // ========================================================================

    static void WriteR16(float[] hmap, string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
                                      bufferSize: 1 << 20);
        using var bw = new BinaryWriter(fs);

        // UE5 Landscape Import expects heightmap[index=0] = landscape vertex (X=0, Y=0),
        // which is the SW corner (lat=LAT_MIN). Internally, py=0 maps to lat=LAT_MAX
        // (north-at-top image convention), so flip Y on write.
        for (int py = MAP_H - 1; py >= 0; py--)
        {
            for (int px = 0; px < MAP_W; px++)
            {
                ushort v = (ushort)Math.Clamp((int)(hmap[py * MAP_W + px] * 65535f), 0, 65535);
                bw.Write(v);
            }
        }
    }

    // ========================================================================
    // Zone mask PNG
    // ========================================================================

    static void WriteZoneMask(OceanZone[] zones, string path)
    {
        using var img = new Image<Rgba32>(MAP_W, MAP_H);
        img.ProcessPixelRows(accessor =>
        {
            for (int py = 0; py < MAP_H; py++)
            {
                var row = accessor.GetRowSpan(py);
                for (int px = 0; px < MAP_W; px++)
                {
                    var (lon, lat) = PixelToLonLat(px, py);
                    row[px] = new Rgba32(40, 40, 40, 255);

                    for (int zi = 0; zi < zones.Length; zi++)
                    {
                        var zone = zones[zi];
                        bool inZone = (zone.LonMin <= zone.LonMax)
                            ? lon >= zone.LonMin && lon <= zone.LonMax
                            : lon >= zone.LonMin || lon <= zone.LonMax;
                        if (!inZone || lat < zone.LatMin || lat > zone.LatMax) continue;

                        var c = ZoneColors[zi % ZoneColors.Length];
                        row[px] = new Rgba32(c.R, c.G, c.B, 180);
                        break;
                    }
                }
            }
        });
        img.SaveAsPng(path);
    }

    // ========================================================================
    // Land mask PNG
    // ========================================================================

    static void WriteLandMask(bool[] isLand, string path)
    {
        using var img = new Image<Rgba32>(MAP_W, MAP_H);
        img.ProcessPixelRows(accessor =>
        {
            for (int py = 0; py < MAP_H; py++)
            {
                var row = accessor.GetRowSpan(py);
                for (int px = 0; px < MAP_W; px++)
                {
                    byte v = isLand[py * MAP_W + px] ? (byte)220 : (byte)30;
                    row[px] = new Rgba32(v, v, v, 255);
                }
            }
        });
        img.SaveAsPng(path);
    }

    // ========================================================================
    // Terrain Splatmap PNG (4-channel layer weights for UE5 Landscape material)
    // ========================================================================

    static void WriteSplatmap(float[] hmap, bool[] isLand, int[] coastDist, string path)
    {
        using var img = new Image<Rgba32>(MAP_W, MAP_H);
        img.ProcessPixelRows(accessor =>
        {
            for (int py = 0; py < MAP_H; py++)
            {
                var row = accessor.GetRowSpan(py);
                for (int px = 0; px < MAP_W; px++)
                {
                    int idx = py * MAP_W + px;
                    float h = hmap[idx];
                    int cd = coastDist[idx];

                    // Ocean pixels: all zero
                    if (h < 0.49f && !(cd >= -2 && cd <= 3))
                    {
                        row[px] = new Rgba32(0, 0, 0, 0);
                        continue;
                    }

                    // R = Beach/Sand: narrow band at coast (h 0.49-0.54, coastDist -2 to +3)
                    float beach = 0f;
                    if (cd >= -2 && cd <= 3)
                    {
                        // Height contribution: ramp up from 0.49, peak at 0.51, ramp down to 0.54
                        float hBeach;
                        if (h < 0.49f)      hBeach = 0f;
                        else if (h < 0.51f) hBeach = (h - 0.49f) / 0.02f;      // 0 -> 1
                        else if (h < 0.54f) hBeach = 1.0f - (h - 0.51f) / 0.03f; // 1 -> 0
                        else                hBeach = 0f;

                        // Distance contribution: strongest at coast (cd 0-1), fade at edges
                        float dBeach;
                        if (cd < -2)      dBeach = 0f;
                        else if (cd < 0)  dBeach = (cd + 2) / 2.0f;   // -2 -> 0: ramp 0->1
                        else if (cd <= 1) dBeach = 1.0f;
                        else if (cd <= 3) dBeach = 1.0f - (cd - 1) / 2.0f; // 1->3: ramp 1->0
                        else              dBeach = 0f;

                        beach = hBeach * dBeach;
                    }

                    // G = Grass/Lowland: h 0.51-0.60 (overlaps Beach ramp-out for smooth blend)
                    float grass = 0f;
                    if (h >= 0.51f && h < 0.60f)
                    {
                        if (h < 0.55f)      grass = (h - 0.51f) / 0.04f;        // ramp in
                        else if (h < 0.58f) grass = 1.0f;                        // full
                        else                grass = 1.0f - (h - 0.58f) / 0.02f;  // ramp out
                    }

                    // B = Rock/Mountain: h 0.58-0.72
                    float rock = 0f;
                    if (h >= 0.58f && h < 0.72f)
                    {
                        if (h < 0.61f)      rock = (h - 0.58f) / 0.03f;         // ramp in
                        else if (h < 0.68f) rock = 1.0f;                         // full
                        else                rock = 1.0f - (h - 0.68f) / 0.04f;   // ramp out
                    }

                    // A = Snow/Peak: h 0.70+
                    float snow = 0f;
                    if (h >= 0.70f)
                    {
                        if (h < 0.76f)  snow = (h - 0.70f) / 0.06f;  // ramp in
                        else            snow = 1.0f;                    // full
                    }

                    // Normalize so channels sum to 255
                    float total = beach + grass + rock + snow;
                    if (total < 0.001f)
                    {
                        // Fallback for land pixels in gaps between layers
                        if (isLand[idx])
                        {
                            if (h < 0.55f) beach = 1f;
                            else           grass = 1f;
                            total = 1f;
                        }
                        else
                        {
                            row[px] = new Rgba32(0, 0, 0, 0);
                            continue;
                        }
                    }

                    float scale = 255.0f / total;
                    byte rB = (byte)Math.Clamp((int)(beach * scale + 0.5f), 0, 255);
                    byte gB = (byte)Math.Clamp((int)(grass * scale + 0.5f), 0, 255);
                    byte bB = (byte)Math.Clamp((int)(rock  * scale + 0.5f), 0, 255);
                    byte aB = (byte)Math.Clamp((int)(snow  * scale + 0.5f), 0, 255);

                    row[px] = new Rgba32(rB, gB, bB, aB);
                }
            }
        });
        img.SaveAsPng(path);
    }

    // ========================================================================
    // Terrain Colormap PNG (stylized Uncharted Waters 2 visualization)
    // ========================================================================

    static void WriteColormap(float[] hmap, bool[] isLand, int[] coastDist, string path)
    {
        // Color stop definitions: (maxH, R, G, B)
        // We lerp between adjacent stops for smooth transitions.
        (float h, byte r, byte g, byte b)[] stops = new[]
        {
            (0.00f, (byte) 15, (byte) 30, (byte) 70),   // Deep ocean floor
            (0.15f, (byte) 15, (byte) 30, (byte) 70),   // Deep ocean
            (0.25f, (byte) 25, (byte) 50, (byte)110),   // Mid ocean
            (0.38f, (byte) 35, (byte) 70, (byte)130),   // Continental slope
            (0.45f, (byte) 50, (byte)100, (byte)160),   // Shallow start
            (0.49f, (byte) 70, (byte)130, (byte)180),   // Shallow end
            (0.51f, (byte)210, (byte)190, (byte)140),   // Beach/coast
            (0.53f, (byte)190, (byte)175, (byte)120),   // Beach end
            (0.54f, (byte)100, (byte)170, (byte) 70),   // Lowland start (bright green)
            (0.56f, (byte) 80, (byte)155, (byte) 55),   // Lowland mid
            (0.58f, (byte) 70, (byte)140, (byte) 50),   // Lowland/hills transition
            (0.61f, (byte)130, (byte)130, (byte) 55),   // Hills (olive)
            (0.65f, (byte)140, (byte)120, (byte) 50),   // Hills/mountain transition
            (0.68f, (byte)120, (byte) 85, (byte) 45),   // Mountain
            (0.72f, (byte)140, (byte)130, (byte)120),   // Mountain/snow transition
            (0.78f, (byte)210, (byte)210, (byte)220),   // Snow peak
            (1.00f, (byte)240, (byte)240, (byte)250),   // Snow peak cap
        };

        using var img = new Image<Rgba32>(MAP_W, MAP_H);
        img.ProcessPixelRows(accessor =>
        {
            for (int py = 0; py < MAP_H; py++)
            {
                var row = accessor.GetRowSpan(py);
                for (int px = 0; px < MAP_W; px++)
                {
                    int idx = py * MAP_W + px;
                    float h = hmap[idx];

                    // Find the two stops to lerp between
                    int si = 0;
                    for (int i = 1; i < stops.Length; i++)
                    {
                        if (stops[i].h >= h) { si = i - 1; break; }
                        si = i - 1;
                    }
                    if (si >= stops.Length - 1) si = stops.Length - 2;

                    var lo = stops[si];
                    var hi = stops[si + 1];

                    float range = hi.h - lo.h;
                    float t = range > 0.0001f ? Math.Clamp((h - lo.h) / range, 0f, 1f) : 0f;

                    byte cr = (byte)Math.Clamp((int)(lo.r + (hi.r - lo.r) * t + 0.5f), 0, 255);
                    byte cg = (byte)Math.Clamp((int)(lo.g + (hi.g - lo.g) * t + 0.5f), 0, 255);
                    byte cb = (byte)Math.Clamp((int)(lo.b + (hi.b - lo.b) * t + 0.5f), 0, 255);

                    row[px] = new Rgba32(cr, cg, cb, 255);
                }
            }
        });
        img.SaveAsPng(path);
    }
}
