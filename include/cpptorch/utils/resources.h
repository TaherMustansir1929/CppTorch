#pragma once

#include <cstdlib>
#include <string>

namespace cpptorch
{

/**
 * Locates the directory that holds the CSV datasets shipped with CppTorch.
 *
 * Lookup order:
 *   1. Directory installed through set_dir().
 *   2. The CPPTORCH_RESOURCES_PATH environment variable.
 *   3. The CPPTORCH_DEFAULT_RESOURCES_PATH compile definition (set automatically
 *      for in-tree builds of the tests and examples).
 *   4. "./resources/" relative to the current working directory.
 */
class Resources
{
public:
    /**
     * Overrides the resources directory. An empty string clears the override.
     */
    static void set_dir(const std::string &dir)
    {
        s_dir = dir;
        if (!s_dir.empty() && s_dir.back() != '/')
            s_dir.push_back('/');
    }

    /**
     * Returns the resources directory, always terminated with a '/'.
     */
    static const std::string &dir()
    {
        if (!s_dir.empty())
            return s_dir;

        if (s_cached.empty())
        {
            std::string candidate;

            const char *env = std::getenv("CPPTORCH_RESOURCES_PATH");
            if (env != nullptr && env[0] != '\0')
            {
                candidate = env;
            }
            else
            {
#ifdef CPPTORCH_DEFAULT_RESOURCES_PATH
                candidate = CPPTORCH_DEFAULT_RESOURCES_PATH;
#else
                candidate = "./resources/";
#endif
            }

            if (!candidate.empty() && candidate.back() != '/')
                candidate.push_back('/');

            s_cached = candidate;
        }

        return s_cached;
    }

    /**
     * Builds the full path of a file inside the resources directory.
     */
    static std::string path(const std::string &filename)
    {
        return dir() + filename;
    }

private:
    inline static std::string s_dir;
    inline static std::string s_cached;
};

} // namespace cpptorch
