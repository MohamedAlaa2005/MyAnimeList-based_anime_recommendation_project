var animelist=''
var anime=0
const addanime=28
const params = new URLSearchParams(window.location.search);
const User_name = params.get('User_name');
const container = document.getElementById('anime-list');
container.innerHTML = ''; 
if (!User_name) {
  console.log("No title provided in URL");

}else{
        console.log(User_name);
}
function fetchAnimeList() {


  fetch(`/userjs?User_name=${encodeURIComponent(User_name)}`)
    .then(response => {
      return response.json();
    })
    .then(data => {
      console.log("Data received:", data);
      displayAnimeList(data); // ✅
    })
    .catch(error => {
      console.error('Error fetching anime:', error);
    })
    .finally(() => {
      const loader = document.querySelector('.loader-container');
      if (loader) loader.style.display = 'none';
    });
}

function displayAnimeList(data) {
  animelist=data
  let html = '';

  for (let index = anime; index < addanime+anime; index++) {
    const encodedTitle = encodeURIComponent(animelist[index].title || 'unknown');
    html += `
      <li>
        <a href="/anime_profile?title=${encodedTitle}" class="a">
          <div>
            <img src="${animelist[index].img_url}" alt="Image not found" width="150">
            <p><strong>${animelist[index].title || 'Unknown Title'}</strong> (${animelist[index].anime_year || 'N/A'})</p>
            <p>⭐ ${animelist[index].score || 'N/A'}</p>
          </div>
        </a>
      </li>
    `;
  }
   anime+=addanime
  container.innerHTML = html;
}

window.addEventListener('load', fetchAnimeList);
let loading = false;

let debounceTimeout;
window.addEventListener('scroll', () => {
  if (debounceTimeout || loading) return;

  debounceTimeout = setTimeout(() => {
    debounceTimeout = null;

    const scrollTop = window.scrollY;
    const windowHeight = window.innerHeight;
    const docHeight = document.documentElement.scrollHeight;

    if (scrollTop + windowHeight >= docHeight - 100) {
      loading = true;

      let html = '';
      for (let index = anime; index < Math.min(anime + addanime, animelist.length); index++) {
        const encodedTitle = encodeURIComponent(animelist[index].title || 'unknown');
        html += `
          <li>
            <a href="/anime_profile?title=${encodedTitle}" class="a">
              <div>
                <img src="${animelist[index].img_url}" alt="Image not found" width="150">
                <p><strong>${animelist[index].title || 'Unknown Title'}</strong> (${animelist[index].anime_year || 'N/A'})</p>
                <p>⭐ ${animelist[index].score || 'N/A'}</p>
              </div>
            </a>
          </li>
        `;
      }

      anime += addanime;
      container.innerHTML += html;

      // Wait a bit before allowing the next scroll-triggered load
      setTimeout(() => loading = false, 300);
    }
  }, 800); // Debounce delay
});
