using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;

namespace DataGenerator
{
    public class DataGeneratorCore
    {
        public Action<string>? OnLog;

        private void Log(string Message)
        {
            OnLog?.Invoke(Message);
        }

        public void Run()
        {
            this.Log("Starting generation process...");

            try
            {
                string? RootPath = FindRootPath(AppDomain.CurrentDomain.BaseDirectory);
                
                if (RootPath == null)
                {
                    this.Log("Error: Could not determine project root (searching for Template directory).");
                    return;
                }

                string TemplatePath = Path.Combine(RootPath, "Template");
                string GameDataPath = Path.Combine(RootPath, "GameData");
                string CodeGenPath = Path.Combine(RootPath, "Source/UnrealWorld/Public/Data/Generated");

                if (!Directory.Exists(TemplatePath))
                {
                    this.Log("Error: Template directory not found at " + TemplatePath);
                    return;
                }

                Directory.CreateDirectory(GameDataPath);
                Directory.CreateDirectory(CodeGenPath);

                string[] MarkdownFiles = Directory.GetFiles(TemplatePath, "*.md");
                foreach (string FilePath in MarkdownFiles)
                {
                    this.ProcessMarkdownFile(FilePath, GameDataPath, CodeGenPath);
                }

                this.Log("Generation completed successfully!");
            }
            catch (Exception ex)
            {
                this.Log("Fatal Error: " + ex.Message);
            }
        }

        private void ProcessMarkdownFile(string FilePath, string GameDataPath, string CodeGenPath)
        {
            string FileName = Path.GetFileNameWithoutExtension(FilePath);
            this.Log("Processing: " + FileName + ".md");

            string[] Lines = File.ReadAllLines(FilePath);
            List<string> TableLines = new List<string>();

            foreach (string Line in Lines)
            {
                string Trimmed = Line.Trim();
                if (Trimmed.StartsWith("|") && Trimmed.EndsWith("|"))
                {
                    TableLines.Add(Trimmed);
                }
            }

            if (TableLines.Count < 3)
            {
                this.Log("Skip: No valid table found in " + FileName);
                return;
            }

            string HeaderLine = TableLines[0];
            string[] Headers = HeaderLine.Split('|', StringSplitOptions.RemoveEmptyEntries).Select(h => h.Trim()).ToArray();

            List<Dictionary<string, object>> JsonEntries = new List<Dictionary<string, object>>();
            for (int i = 2; i < TableLines.Count; i++)
            {
                string[] Cells = TableLines[i].Split('|', StringSplitOptions.RemoveEmptyEntries).Select(c => c.Trim()).ToArray();
                if (Cells.Length != Headers.Length) continue;

                Dictionary<string, object> Row = new Dictionary<string, object>();
                for (int j = 0; j < Headers.Length; j++)
                {
                    string[] TypeAndName = Headers[j].Split(':');
                    if (TypeAndName.Length < 2) continue;

                    string Type = TypeAndName[0].ToLower();
                    string Name = TypeAndName[1];
                    string Value = Cells[j];

                    try
                    {
                        if (Type == "int") Row[Name] = int.Parse(Value);
                        else if (Type == "float") Row[Name] = float.Parse(Value);
                        else if (Type == "bool") Row[Name] = bool.Parse(Value);
                        else Row[Name] = Value;
                    }
                    catch (Exception)
                    {
                        this.Log("Warning: Failed to parse '" + Value + "' as " + Type + " in " + FileName + " at row " + i.ToString());
                        Row[Name] = Value;
                    }
                }
                JsonEntries.Add(Row);
            }

            string JsonOutput = JsonConvert.SerializeObject(JsonEntries, Formatting.Indented);
            File.WriteAllText(Path.Combine(GameDataPath, FileName + ".json"), JsonOutput);
            this.Log("Saved JSON: " + FileName + ".json");

            this.GenerateCppHeader(FileName, Headers, CodeGenPath);
        }

        private void GenerateCppHeader(string FileName, string[] Headers, string CodeGenPath)
        {
            StringBuilder CppCode = new StringBuilder();
            CppCode.AppendLine("#pragma once");
            CppCode.AppendLine("");
            CppCode.AppendLine("#include \"CoreMinimal.h\"");
            CppCode.AppendLine("#include \"Engine/DataTable.h\"");
            CppCode.AppendLine("#include \"" + FileName + "Generated.generated.h\"");
            CppCode.AppendLine("");
            CppCode.AppendLine("USTRUCT(BlueprintType)");
            CppCode.AppendLine("struct F" + FileName + " : public FTableRowBase");
            CppCode.AppendLine("{");
            CppCode.AppendLine("\tGENERATED_BODY()");
            CppCode.AppendLine("");

            foreach (string Header in Headers)
            {
                string[] TypeAndName = Header.Split(':');
                if (TypeAndName.Length < 2) continue;

                string Type = TypeAndName[0].ToLower();
                string Name = TypeAndName[1];

                string UnrealType = "FString";
                if (Type == "int") UnrealType = "int32";
                else if (Type == "float") UnrealType = "float";
                else if (Type == "bool") UnrealType = "bool";
                else if (Type == "fname") UnrealType = "FName";

                CppCode.AppendLine("\tUPROPERTY(EditAnywhere, BlueprintReadWrite, Category = \"Data\")");
                CppCode.AppendLine("\t" + UnrealType + " " + Name + ";");
                CppCode.AppendLine("");
            }

            CppCode.AppendLine("};");

            File.WriteAllText(Path.Combine(CodeGenPath, FileName + "Generated.h"), CppCode.ToString());
            this.Log("Generated C++: " + FileName + "Generated.h");
        }

        private string? FindRootPath(string StartPath)
        {
            string? Current = StartPath;
            while (Current != null)
            {
                if (Directory.Exists(Path.Combine(Current, "Template")) && 
                    Directory.Exists(Path.Combine(Current, "Source")))
                {
                    return Current;
                }
                Current = Directory.GetParent(Current)?.FullName;
            }
            return null;
        }
    }
}
