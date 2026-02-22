#pragma once

#include "Common.h"
#include <iostream>
#include <sstream>

/**
 * @brief Lightweight logging system
 * 
 * Outputs to console in debug builds and supports OutputDebugString,
 * with minimal overhead in release builds.
 */
class KLogger
{
public:
    enum class ELevel
    {
        Info,
        Warning,
        Error
    };

    /**
     * @brief Output info log
     * @param Message Output message
     */
    static void Info(const std::string& Message)
    {
        Log(ELevel::Info, Message);
    }

    /**
     * @brief Output warning log
     * @param Message Output message
     */
    static void Warning(const std::string& Message)
    {
        Log(ELevel::Warning, Message);
    }

    /**
     * @brief Output error log
     * @param Message Output message
     */
    static void Error(const std::string& Message)
    {
        Log(ELevel::Error, Message);
    }

    /**
     * @brief Output HRESULT error log
     * @param Result HRESULT value
     * @param Context Error context
     */
    static void HResultError(HRESULT Result, const std::string& Context)
    {
        std::ostringstream oss;
        oss << Context << " - HRESULT: 0x" << std::hex << Result;
        Error(oss.str());
    }

private:
    static void Log(ELevel Level, const std::string& Message)
    {
#ifdef _DEBUG
        std::string Prefix;
        switch (Level)
        {
        case ELevel::Info:    Prefix = "[INFO] "; break;
        case ELevel::Warning: Prefix = "[WARN] "; break;
        case ELevel::Error:   Prefix = "[ERROR] "; break;
        }

        std::string FullMessage = Prefix + Message + "\n";
        
        // Console output
        std::cout << FullMessage;
        
        // Output to Visual Studio output window
        OutputDebugStringA(FullMessage.c_str());
#endif
    }
}; 